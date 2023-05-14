#include "ggml/ggml.h"

#include "common.h"
#include "common-ggml.h"

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <utility>
#include <vector>
#include <iostream>
#include <unistd.h>
#include <sys/mman.h>

// default hparams (GPT-J 6B)
struct gptj_hparams {
    uint32_t n_vocab = 50400;
    uint32_t n_ctx = 2048;
    uint32_t n_embd = 4096;
    uint32_t n_head = 16;
    uint32_t n_layer = 28;
    uint32_t n_rot = 64;
    uint32_t ftype = 1;
};

static std::string format(const char *fmt, ...) {
    va_list ap, ap2;
    va_start(ap, fmt);
    va_copy(ap2, ap);
    int size = vsnprintf(NULL, 0, fmt, ap);
    GGML_ASSERT(size >= 0 && size < INT_MAX);
    std::vector<char> buf(size + 1);
    int size2 = vsnprintf(buf.data(), size + 1, fmt, ap2);
    GGML_ASSERT(size2 == size);
    va_end(ap2);
    va_end(ap);
    return std::string(buf.data(), size);
}

struct gptj_file {
    // use FILE * so we don't have to re-open the file to mmap
    FILE *fp;
    size_t file_size;

    gptj_file(const char *fname, const char *mode) {
        fp = std::fopen(fname, mode);
        if (fp == NULL) {
            throw std::runtime_error(format("failed to open %s: %s", fname, strerror(errno)));
        }
        seek(0, SEEK_END);
        file_size = tell();
        seek(0, SEEK_SET);
    }

    size_t tell() const {
        long ret = std::ftell(fp);
        GGML_ASSERT(ret != -1); // this really shouldn't fail
        return (size_t) ret;
    }

    void seek(size_t offset, int whence) const {
        int ret = std::fseek(fp, (long) offset, whence);
        GGML_ASSERT(ret == 0);
    }

    void read_raw(void *ptr, size_t size) const {
        if (size == 0) {
            return;
        }
        errno = 0;
        std::size_t ret = std::fread(ptr, size, 1, fp);
        if (ferror(fp)) {
            throw std::runtime_error(format("read error: %s", strerror(errno)));
        }
        if (ret != 1) {
            throw std::runtime_error(std::string("unexpectedly reached end of file"));
        }
    }

    std::uint32_t read_u32() const {
        std::uint32_t ret;
        read_raw(&ret, sizeof(ret));
        return ret;
    }

    std::string read_string(std::uint32_t len) const {
        std::vector<char> chars(len);
        read_raw(chars.data(), len);
        return std::string(chars.data(), len);
    }

    ~gptj_file() {
        if (fp) {
            std::fclose(fp);
        }
    }
};


template<typename T>
static T checked_mul(T a, T b) {
    T ret = a * b;
    if (a != 0 && ret / a != b) {
        throw format("overflow multiplying %llu * %llu",
                     (unsigned long long) a, (unsigned long long) b);
    }
    return ret;
}


static size_t calc_tensor_size(const std::vector<uint32_t> &ne, enum ggml_type type) {
    size_t size = ggml_type_size(type);
    for (uint32_t dim: ne) {
        size = checked_mul<size_t>(size, dim);
    }
    return size / ggml_blck_size(type);
}

struct load_tensor {
    std::string name;
    enum ggml_type type;
    std::vector<uint32_t> ne;
    size_t size;
    size_t file_offset;
    struct ggml_tensor *ggml_tensor = NULL;
    uint8_t *data;

    load_tensor(std::string name, enum ggml_type type, std::vector<uint32_t> ne, size_t file_offset) :
            name(std::move(name)),
            type(type),
            ne(std::move(ne)),
            size(calc_tensor_size(this->ne, this->type)),
            file_offset(file_offset) {
    }
};


struct load_tensors_map {
    // tensors is kept in a separate vector to preserve file order
    std::vector<load_tensor> tensors;
    std::unordered_map<std::string, size_t> name_to_idx;

    void add_tensor(load_tensor load_tensor) {
        if (name_to_idx.find(load_tensor.name) != name_to_idx.end()) {
            throw format("tensor with name %s is already loaded\n", load_tensor.name.c_str());
        }

        name_to_idx.emplace(load_tensor.name, tensors.size());
        tensors.emplace_back(std::move(load_tensor));
    }
};

struct gptj_file_loader {
    gptj_file file;
    gptj_hparams hparams;
    gpt_vocab vocab;

    gptj_file_loader(const char *fname, load_tensors_map &tensors_map)
            : file(fname, "rb") {
        fprintf(stderr, "loading model from %s\n", fname);
        read_magic();
        read_hparams();
        read_vocab();
        read_tensor_metadata(tensors_map);
    }

    void read_magic() const {
        uint32_t magic = file.read_u32();
        GGML_ASSERT(magic == 0x67676d6c);
    }

    void read_hparams() {
        hparams.n_vocab = file.read_u32();
        hparams.n_ctx = file.read_u32();
        hparams.n_embd = file.read_u32();
        hparams.n_head = file.read_u32();
        hparams.n_layer = file.read_u32();
        hparams.n_rot = file.read_u32();
        hparams.ftype = file.read_u32();

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: n_rot   = %d\n", __func__, hparams.n_rot);
        printf("%s: ftype   = %d\n", __func__, hparams.ftype);
    }

    void read_vocab() {
        uint32_t n_vocab = file.read_u32();
        GGML_ASSERT(n_vocab == hparams.n_vocab);

        for (int i = 0; i < n_vocab; i++) {
            uint32_t len = file.read_u32();
            std::string word = file.read_string(len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
        printf("%s: vocab size = %zu\n", __func__, vocab.id_to_token.size());
    }

    void read_tensor_metadata(load_tensors_map &tensors_map) const {
        while (file.tell() < file.file_size) {
//            printf("%s: position = %zu, file size = %zu\n", __func__, file.tell(), file.file_size);
            uint32_t n_dims = file.read_u32();
            uint32_t name_len = file.read_u32();
            auto ttype = (enum ggml_type) file.read_u32();
//            printf("%s: ndims = %u, name_len = %u, ttype = %u\n", __func__, n_dims, name_len, ttype);
            std::vector<uint32_t> ne(n_dims);
            file.read_raw(ne.data(), sizeof(ne[0]) * n_dims);
//            printf("%s: position = %zu, ne size = %zu\n", __func__, file.tell(), ne.size());
            std::string name = file.read_string(name_len);
//            printf("%s: name = %s\n", __func__, name.c_str());
            if (n_dims < 1 || n_dims > 2) {
                throw format("llama.cpp: tensor '%s' should not be %u-dimensional", name.c_str(), n_dims);
            }
            switch (ttype) {
                case GGML_TYPE_F32:
                case GGML_TYPE_F16:
                case GGML_TYPE_Q4_0:
                case GGML_TYPE_Q4_1:
                case GGML_TYPE_Q5_0:
                case GGML_TYPE_Q5_1:
                case GGML_TYPE_Q8_0:
                    break;
                default: {
                    throw format("unrecognized tensor type %u\n", ttype);
                }
            }

            size_t file_offset = file.tell();
            load_tensor load_tensor(name, ttype, ne, file_offset);
//            printf("%s: load tensor size = %zu\n", __func__, load_tensor.size);
            file.seek(load_tensor.size, SEEK_CUR);

            tensors_map.add_tensor(load_tensor);
        }
    }
};

struct gptj_mmap {
    void *addr;
    size_t size;

    gptj_mmap(const gptj_mmap &) = delete;

    explicit gptj_mmap(struct gptj_file *file) {
        size = file->file_size;
        int fd = fileno(file->fp);
        addr = mmap(NULL, file->file_size, PROT_READ, MAP_SHARED, fd, 0);
        if (addr == MAP_FAILED) {
            throw std::runtime_error(format("mmap failed: %s", strerror(errno)));
        }

        // Advise the kernel to preload the mapped memory
        if (madvise(addr, file->file_size, MADV_WILLNEED)) {
            fprintf(stderr, "warning: madvise(.., MADV_WILLNEED) failed: %s\n",
                    strerror(errno));
        }
    }

    ~gptj_mmap() {
        munmap(addr, size);
    }
};

// Represents some region of memory being locked using mlock or VirtualLock;
// will automatically unlock on destruction.
struct gptj_mlock {
    void *lock_addr = NULL;
    size_t lock_size = 0;
    bool failed_already = false;

    gptj_mlock() = default;

    gptj_mlock(const gptj_mlock &) = delete;

    ~gptj_mlock() {
        if (lock_size) {
            raw_unlock(lock_addr, lock_size);
        }
    }

    void init(void *addr) {
        GGML_ASSERT(this->lock_addr == NULL && this->lock_size == 0);
        this->lock_addr = addr;
    }

    void grow_to(size_t target_size) {
        GGML_ASSERT(lock_addr);
        if (failed_already) {
            return;
        }
        size_t granularity = lock_granularity();
        target_size = (target_size + granularity - 1) & ~(granularity - 1);
        if (target_size > lock_size) {
            if (raw_lock((uint8_t *) lock_addr + lock_size, target_size - lock_size)) {
                lock_size = target_size;
            } else {
                failed_already = true;
            }
        }
    }

    static size_t lock_granularity() {
        return (size_t) sysconf(_SC_PAGESIZE);
    }

#ifdef __APPLE__
#define MLOCK_SUGGESTION \
            "Try increasing the sysctl values 'vm.user_wire_limit' and 'vm.global_user_wire_limit' and/or " \
            "decreasing 'vm.global_no_user_wire_amount'.  Also try increasing RLIMIT_MLOCK (ulimit -l).\n"
#else
#define MLOCK_SUGGESTION \
            "Try increasing RLIMIT_MLOCK ('ulimit -l' as root).\n"
#endif

    bool raw_lock(const void *addr, size_t size) {
        if (!mlock(addr, size)) {
            return true;
        } else {
            char *errmsg = std::strerror(errno);
            bool suggest = (errno == ENOMEM);

            // Check if the resource limit is fine after all
            struct rlimit lock_limit;
            if (suggest && getrlimit(RLIMIT_MEMLOCK, &lock_limit))
                suggest = false;
            if (suggest && (lock_limit.rlim_max > lock_limit.rlim_cur + size))
                suggest = false;

            fprintf(stderr, "warning: failed to mlock %zu-byte buffer (after previously locking %zu bytes): %s\n%s",
                    size, this->lock_size, errmsg, suggest ? MLOCK_SUGGESTION : "");
            return false;
        }
    }

#undef MLOCK_SUGGESTION

    void raw_unlock(void *addr, size_t size) {
        if (munlock(addr, size)) {
            fprintf(stderr, "warning: failed to munlock buffer: %s\n", std::strerror(errno));
        }
    }
};


static std::string format_tensor_shape(const std::vector<uint32_t> &ne) {
    char buf[256];
    snprintf(buf, sizeof(buf), "%5u", ne.at(0));
    for (size_t i = 1; i < ne.size(); i++) {
        snprintf(buf + strlen(buf), sizeof(buf) - strlen(buf), " x %5u", ne.at(i));
    }
    return buf;
}


struct gptj_model_loader {
    load_tensors_map tensors_map;
    gptj_file_loader file_loader;
    size_t num_ggml_tensors_created = 0;
    struct ggml_context *ggml_ctx = NULL;
    std::unique_ptr<gptj_mmap> mapping;

    explicit gptj_model_loader(const std::string &fname) :
            file_loader(fname.c_str(), tensors_map) {
    }

    void calc_sizes(size_t *ctx_size_p, size_t *mmapped_size_p) const {
        *ctx_size_p = *mmapped_size_p = 0;
        for (const load_tensor &lt: tensors_map.tensors) {
            *ctx_size_p += sizeof(struct ggml_tensor) + GGML_OBJECT_SIZE;
            *mmapped_size_p += lt.size;
        }
    }

    struct ggml_tensor *get_tensor(const std::string &name, const std::vector<uint32_t> &ne, enum ggml_type ttype) {
        auto it = tensors_map.name_to_idx.find(name);
        if (it == tensors_map.name_to_idx.end()) {
            throw std::runtime_error(format("llama.cpp: tensor '%s' is missing from model", name.c_str()));
        }
        load_tensor &lt = tensors_map.tensors.at(it->second);
        if (lt.ne != ne) {
            throw std::runtime_error(format("llama.cpp: tensor '%s' has wrong shape; expected %s, got %s",
                                            name.c_str(), format_tensor_shape(ne).c_str(),
                                            format_tensor_shape(lt.ne).c_str()));
        }
        if (lt.type != ttype) {
            throw std::runtime_error(format("llama.cpp: tensor '%s' has wrong type; expected %d, got %d",
                                            name.c_str(), ttype, lt.type));
        }

        return get_tensor_for(lt);
    }

    struct ggml_tensor *get_tensor_for(load_tensor &lt) {
        struct ggml_tensor *tensor;
        if (lt.ne.size() == 2) {
            tensor = ggml_new_tensor_2d(ggml_ctx, lt.type, lt.ne.at(0), lt.ne.at(1));
        } else {
            GGML_ASSERT(lt.ne.size() == 1);
            tensor = ggml_new_tensor_1d(ggml_ctx, lt.type, lt.ne.at(0));
        }
        ggml_set_name(tensor, lt.name.c_str());
        GGML_ASSERT(lt.ggml_tensor == NULL); // if this fails, we called get_tensor twice on the same tensor
        lt.ggml_tensor = tensor;
        num_ggml_tensors_created++;
        return tensor;
    }

    void done_getting_tensors() const {
        if (num_ggml_tensors_created != tensors_map.tensors.size()) {
            throw std::runtime_error(std::string("file contained more tensors than expected"));
        }
    }

    void load_all_data(gptj_mlock *lmlock) {
        size_t data_size = 0;
        for (const load_tensor &lt: tensors_map.tensors) {
            data_size += lt.size;
        }

        mapping.reset(new gptj_mmap(&file_loader.file));
        if (lmlock) {
            lmlock->init(mapping->addr);
        }

        size_t done_size = 0;
        for (load_tensor &lt: tensors_map.tensors) {
            printf(".");
            fflush(stdout);
            GGML_ASSERT(lt.ggml_tensor); // unused tensors should have been caught by load_data already
            lt.data = (uint8_t *) mapping->addr + lt.file_offset;
//            print_checksum(lt);
            lt.ggml_tensor->data = lt.data;
            done_size += lt.size;
            if (lmlock) {
                lmlock->grow_to(done_size);
            }
        }
        printf("\n");
        fflush(stdout);
    }

    static void print_checksum(load_tensor &lt) {
        uint32_t sum = 0;
        for (size_t i = 0; i < lt.size; i++) {
            uint8_t byte = lt.data[i];
            sum = byte + (sum << 6) + (sum << 16) - sum; // sdbm hash
        }
        printf("%s checksum: %#08x (%s, size %zu)\n", lt.name.c_str(), sum,
               format_tensor_shape(lt.ne).c_str(), lt.size);
    }
};


struct gptj_layer {
    // normalization
    struct ggml_tensor *ln_1_g;
    struct ggml_tensor *ln_1_b;

    // attention
    struct ggml_tensor *c_attn_q_proj_w;
    struct ggml_tensor *c_attn_k_proj_w;
    struct ggml_tensor *c_attn_v_proj_w;

    struct ggml_tensor *c_attn_proj_w;

    // ff
    struct ggml_tensor *c_mlp_fc_w;
    struct ggml_tensor *c_mlp_fc_b;

    struct ggml_tensor *c_mlp_proj_w;
    struct ggml_tensor *c_mlp_proj_b;
};


// Replacement for std::vector<uint8_t> that doesn't require zero-initialization.
struct gptj_buffer {
    uint8_t *addr = NULL;
    size_t size = 0;

    gptj_buffer() = default;

    void resize(size_t size) {
        delete[] addr;
        addr = new uint8_t[size];
        this->size = size;
    }

    ~gptj_buffer() {
        delete[] addr;
    }

    // disable copy and move
    gptj_buffer(const gptj_buffer &) = delete;

    gptj_buffer(gptj_buffer &&) = delete;

    gptj_buffer &operator=(const gptj_buffer &) = delete;

    gptj_buffer &operator=(gptj_buffer &&) = delete;
};

struct kv_cache {
    struct ggml_tensor *k;
    struct ggml_tensor *v;

    struct ggml_context *ctx = NULL;

    gptj_buffer buf;

    ~kv_cache() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

struct gptj_model {
    gptj_hparams hparams;

    // normalization
    struct ggml_tensor *ln_f_g;
    struct ggml_tensor *ln_f_b;

    struct ggml_tensor *wte; // position embedding

    struct ggml_tensor *lmh_g; // language model head
    struct ggml_tensor *lmh_b; // language model bias

    std::vector<gptj_layer> layers;

    struct ggml_context *ctx;
    gptj_buffer buf;
    std::unique_ptr<gptj_mmap> mapping;
    std::map<std::string, struct ggml_tensor *> tensors;

    struct kv_cache kv_self;

    ~gptj_model() {
        if (ctx) {
            ggml_free(ctx);
        }
    }
};

static bool kv_cache_init(const struct gptj_hparams &hparams, struct kv_cache &cache) {
    const uint32_t n_embd = hparams.n_embd;
    const uint32_t n_layer = hparams.n_layer;
    const uint32_t n_ctx = hparams.n_ctx;

    const int64_t n_mem = n_layer * n_ctx;
    const int64_t n_elements = n_embd * n_mem;

    cache.buf.resize(2u * n_elements * ggml_type_size(GGML_TYPE_F16) + 2u * 1024 * 1024);

    struct ggml_init_params params = {
            .mem_size   = cache.buf.size,
            .mem_buffer = cache.buf.addr,
            .no_alloc   = false,
    };

    cache.ctx = ggml_init(params);

    if (!cache.ctx) {
        fprintf(stderr, "%s: failed to allocate memory for kv cache\n", __func__);
        return false;
    }

    cache.k = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F16, n_elements);
    cache.v = ggml_new_tensor_1d(cache.ctx, GGML_TYPE_F16, n_elements);
    ggml_set_name(cache.k, "cache_k");
    ggml_set_name(cache.v, "cache_v");

    return true;
}

bool gptj_model_load_mmap(const std::string &fname, gptj_model &model, gpt_vocab &vocab, gptj_mlock &mlock_mmap) {
    printf("%s: mmap loading model from '%s' - please wait ...\n", __func__, fname.c_str());

    std::unique_ptr<gptj_model_loader> ml(new gptj_model_loader(fname));
    model.hparams = ml->file_loader.hparams;
    vocab = ml->file_loader.vocab;

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype) (model.hparams.ftype));
    printf("%s: wtype = %d\n", __func__, wtype);
    if (wtype == GGML_TYPE_COUNT) {
        fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n",
                __func__, fname.c_str(), model.hparams.ftype);
        return false;
    }

    size_t ctx_size;
    size_t mmapped_size;
    ml->calc_sizes(&ctx_size, &mmapped_size);
    fprintf(stderr, "%s: ggml ctx size = %6.2f KB, mmaped_size = %6.2f MB\n", __func__, ctx_size / 1024.0,
            mmapped_size / (1024.0 * 1024.0));

    // create the ggml context
    {
        model.buf.resize(ctx_size);
        struct ggml_init_params params = {
                .mem_size   = model.buf.size,
                .mem_buffer = model.buf.addr,
                .no_alloc   = true,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        auto &ctx = model.ctx;
        ml->ggml_ctx = ctx;
        const auto &hparams = model.hparams;

        const uint32_t n_layer = hparams.n_layer;
        const uint32_t n_embd = hparams.n_embd;
        const uint32_t n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);
        model.wte = ml->get_tensor("transformer.wte.weight", {n_embd, n_vocab}, wtype);
        model.ln_f_g = ml->get_tensor("transformer.ln_f.weight", {n_embd}, GGML_TYPE_F32);
        model.ln_f_b = ml->get_tensor("transformer.ln_f.bias", {n_embd}, GGML_TYPE_F32);

        model.lmh_g = ml->get_tensor("lm_head.weight", {n_embd, n_vocab}, wtype);
        model.lmh_b = ml->get_tensor("lm_head.bias", {n_vocab}, GGML_TYPE_F32);

        for (int i = 0; i < n_layer; ++i) {
            auto &layer = model.layers[i];
            std::string layers_i = "transformer.h." + std::to_string(i);

            layer.ln_1_g = ml->get_tensor(layers_i + ".ln_1.weight", {n_embd}, GGML_TYPE_F32);
            layer.ln_1_b = ml->get_tensor(layers_i + ".ln_1.bias", {n_embd}, GGML_TYPE_F32);

            layer.c_attn_q_proj_w = ml->get_tensor(layers_i + ".attn.q_proj.weight", {n_embd, n_embd}, wtype);
            layer.c_attn_k_proj_w = ml->get_tensor(layers_i + ".attn.k_proj.weight", {n_embd, n_embd}, wtype);
            layer.c_attn_v_proj_w = ml->get_tensor(layers_i + ".attn.v_proj.weight", {n_embd, n_embd}, wtype);

            layer.c_attn_proj_w = ml->get_tensor(layers_i + ".attn.out_proj.weight", {n_embd, n_embd}, wtype);

            layer.c_mlp_fc_w = ml->get_tensor(layers_i + ".mlp.fc_in.weight", {n_embd, 4 * n_embd}, wtype);
            layer.c_mlp_fc_b = ml->get_tensor(layers_i + ".mlp.fc_in.bias", {4 * n_embd}, GGML_TYPE_F32);

            layer.c_mlp_proj_w = ml->get_tensor(layers_i + ".mlp.fc_out.weight", {4 * n_embd, n_embd}, wtype);
            layer.c_mlp_proj_b = ml->get_tensor(layers_i + ".mlp.fc_out.bias", {n_embd}, GGML_TYPE_F32);
        }
    }


    ml->done_getting_tensors();

    // populate `tensors_by_name`
    for (load_tensor &lt: ml->tensors_map.tensors) {
        model.tensors.emplace(lt.name, lt.ggml_tensor);
    }

    ml->load_all_data(&mlock_mmap);

    model.mapping = std::move(ml->mapping);

    return true;
}

// load the model's weights from a file
bool gptj_model_load(const std::string &fname, gptj_model &model, gpt_vocab &vocab) {
    printf("%s: loading model from '%s' - please wait ...\n", __func__, fname.c_str());

    auto fin = std::ifstream(fname, std::ios::binary);
    if (!fin) {
        fprintf(stderr, "%s: failed to open '%s'\n", __func__, fname.c_str());
        return false;
    }

    // verify magic
    {
        uint32_t magic;
        fin.read((char *) &magic, sizeof(magic));
        if (magic != 0x67676d6c) {
            fprintf(stderr, "%s: invalid model file '%s' (bad magic)\n", __func__, fname.c_str());
            return false;
        }
    }

    // load hparams
    {
        auto &hparams = model.hparams;

        fin.read((char *) &hparams.n_vocab, sizeof(hparams.n_vocab));
        fin.read((char *) &hparams.n_ctx, sizeof(hparams.n_ctx));
        fin.read((char *) &hparams.n_embd, sizeof(hparams.n_embd));
        fin.read((char *) &hparams.n_head, sizeof(hparams.n_head));
        fin.read((char *) &hparams.n_layer, sizeof(hparams.n_layer));
        fin.read((char *) &hparams.n_rot, sizeof(hparams.n_rot));
        fin.read((char *) &hparams.ftype, sizeof(hparams.ftype));

        printf("%s: n_vocab = %d\n", __func__, hparams.n_vocab);
        printf("%s: n_ctx   = %d\n", __func__, hparams.n_ctx);
        printf("%s: n_embd  = %d\n", __func__, hparams.n_embd);
        printf("%s: n_head  = %d\n", __func__, hparams.n_head);
        printf("%s: n_layer = %d\n", __func__, hparams.n_layer);
        printf("%s: n_rot   = %d\n", __func__, hparams.n_rot);
        printf("%s: ftype   = %d\n", __func__, hparams.ftype);
    }

    // load vocab
    {
        uint32_t n_vocab = 0;
        fin.read((char *) &n_vocab, sizeof(n_vocab));

        if (n_vocab != model.hparams.n_vocab) {
            fprintf(stderr, "%s: invalid model file '%s' (bad vocab size %d != %d)\n",
                    __func__, fname.c_str(), n_vocab, model.hparams.n_vocab);
            return false;
        }

        std::string word;
        for (int i = 0; i < n_vocab; i++) {
            uint32_t len;
            fin.read((char *) &len, sizeof(len));

            word.resize(len);
            fin.read((char *) word.data(), len);

            vocab.token_to_id[word] = i;
            vocab.id_to_token[i] = word;
        }
    }

    // for the big tensors, we have the option to store the data in 16-bit floats or quantized
    // in order to save memory and also to speed up the computation
    ggml_type wtype = ggml_ftype_to_ggml_type((ggml_ftype) (model.hparams.ftype));
    if (wtype == GGML_TYPE_COUNT) {
        fprintf(stderr, "%s: invalid model file '%s' (bad ftype value %d)\n",
                __func__, fname.c_str(), model.hparams.ftype);
        return false;
    }

    auto &ctx = model.ctx;

    size_t ctx_size = 0;

    {
        const auto &hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        ctx_size += n_embd * ggml_type_sizef(GGML_TYPE_F32); // ln_f_g
        ctx_size += n_embd * ggml_type_sizef(GGML_TYPE_F32); // ln_f_b

        ctx_size += n_embd * n_vocab * ggml_type_sizef(wtype); // wte

        ctx_size += n_embd * n_vocab * ggml_type_sizef(wtype);         // lmh_g
        ctx_size += n_vocab * ggml_type_sizef(GGML_TYPE_F32); // lmh_b

        ctx_size += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ln_1_g
        ctx_size += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // ln_1_b

        ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // c_attn_q_proj_w
        ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // c_attn_k_proj_w
        ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // c_attn_v_proj_w

        ctx_size += n_layer * (n_embd * n_embd * ggml_type_sizef(wtype)); // c_attn_proj_w

        ctx_size += n_layer * (4 * n_embd * n_embd * ggml_type_sizef(wtype));         // c_mlp_fc_w
        ctx_size += n_layer * (4 * n_embd * ggml_type_sizef(GGML_TYPE_F32)); // c_mlp_fc_b

        ctx_size += n_layer * (4 * n_embd * n_embd * ggml_type_sizef(wtype));         // c_mlp_proj_w
        ctx_size += n_layer * (n_embd * ggml_type_sizef(GGML_TYPE_F32)); // c_mlp_proj_b

        ctx_size += n_ctx * n_layer * n_embd * ggml_type_sizef(GGML_TYPE_F16); // memory_k
        ctx_size += n_ctx * n_layer * n_embd * ggml_type_sizef(GGML_TYPE_F16); // memory_v

        ctx_size += (5 + 10 * n_layer) * 256; // object overhead

        printf("%s: ggml ctx size = %6.2f MB\n", __func__, ctx_size / (1024.0 * 1024.0));
    }

    // create the ggml context
    {
        struct ggml_init_params params = {
                .mem_size   = ctx_size,
                .mem_buffer = NULL,
                .no_alloc   = false,
        };

        model.ctx = ggml_init(params);
        if (!model.ctx) {
            fprintf(stderr, "%s: ggml_init() failed\n", __func__);
            return false;
        }
    }

    // prepare memory for the weights
    {
        const auto &hparams = model.hparams;

        const int n_embd = hparams.n_embd;
        const int n_layer = hparams.n_layer;
        const int n_ctx = hparams.n_ctx;
        const int n_vocab = hparams.n_vocab;

        model.layers.resize(n_layer);

        model.wte = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);

        model.ln_f_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
        model.ln_f_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

        model.lmh_g = ggml_new_tensor_2d(ctx, wtype, n_embd, n_vocab);
        model.lmh_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_vocab);

        // map by name
        model.tensors["transformer.wte.weight"] = model.wte;

        model.tensors["transformer.ln_f.weight"] = model.ln_f_g;
        model.tensors["transformer.ln_f.bias"] = model.ln_f_b;

        model.tensors["lm_head.weight"] = model.lmh_g;
        model.tensors["lm_head.bias"] = model.lmh_b;

        for (int i = 0; i < n_layer; ++i) {
            auto &layer = model.layers[i];

            layer.ln_1_g = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);
            layer.ln_1_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            layer.c_attn_q_proj_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.c_attn_k_proj_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);
            layer.c_attn_v_proj_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);

            layer.c_attn_proj_w = ggml_new_tensor_2d(ctx, wtype, n_embd, n_embd);

            layer.c_mlp_fc_w = ggml_new_tensor_2d(ctx, wtype, n_embd, 4 * n_embd);
            layer.c_mlp_fc_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 4 * n_embd);

            layer.c_mlp_proj_w = ggml_new_tensor_2d(ctx, wtype, 4 * n_embd, n_embd);
            layer.c_mlp_proj_b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, n_embd);

            // map by name
            model.tensors["transformer.h." + std::to_string(i) + ".ln_1.weight"] = layer.ln_1_g;
            model.tensors["transformer.h." + std::to_string(i) + ".ln_1.bias"] = layer.ln_1_b;

            model.tensors["transformer.h." + std::to_string(i) + ".attn.q_proj.weight"] = layer.c_attn_q_proj_w;
            model.tensors["transformer.h." + std::to_string(i) + ".attn.k_proj.weight"] = layer.c_attn_k_proj_w;
            model.tensors["transformer.h." + std::to_string(i) + ".attn.v_proj.weight"] = layer.c_attn_v_proj_w;

            model.tensors["transformer.h." + std::to_string(i) + ".attn.out_proj.weight"] = layer.c_attn_proj_w;

            model.tensors["transformer.h." + std::to_string(i) + ".mlp.fc_in.weight"] = layer.c_mlp_fc_w;
            model.tensors["transformer.h." + std::to_string(i) + ".mlp.fc_in.bias"] = layer.c_mlp_fc_b;

            model.tensors["transformer.h." + std::to_string(i) + ".mlp.fc_out.weight"] = layer.c_mlp_proj_w;
            model.tensors["transformer.h." + std::to_string(i) + ".mlp.fc_out.bias"] = layer.c_mlp_proj_b;
        }
    }

    // key + value memory
    {
//        const auto &hparams = model.hparams;
//
//        const int n_embd = hparams.n_embd;
//        const int n_layer = hparams.n_layer;
//        const int n_ctx = hparams.n_ctx;
//
//        const int n_mem = n_layer * n_ctx;
//        const int n_elements = n_embd * n_mem;

//        model.memory_k = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elements);
//        model.memory_v = ggml_new_tensor_1d(ctx, GGML_TYPE_F16, n_elements);
//
//        const size_t memory_size = ggml_nbytes(model.memory_k) + ggml_nbytes(model.memory_v);

//        printf("%s: memory_size = %8.2f MB, n_mem = %d\n", __func__, memory_size / 1024.0 / 1024.0, n_mem);
    }

    // load weights
    {
        int n_tensors = 0;
        size_t total_size = 0;

        printf("%s: ", __func__);

        while (true) {
            int32_t n_dims;
            int32_t length;
            int32_t ttype;

            fin.read(reinterpret_cast<char *>(&n_dims), sizeof(n_dims));
            fin.read(reinterpret_cast<char *>(&length), sizeof(length));
            fin.read(reinterpret_cast<char *>(&ttype), sizeof(ttype));

//            printf("%s: ndims = %u, name_len = %u, ttype = %u\n", __func__, n_dims, length, ttype);

            if (fin.eof()) {
                break;
            }

            int32_t nelements = 1;
            int32_t ne[2] = {1, 1};
            for (int i = 0; i < n_dims; ++i) {
                fin.read(reinterpret_cast<char *>(&ne[i]), sizeof(ne[i]));
                nelements *= ne[i];
            }

            std::string name(length, 0);
            fin.read(&name[0], length);

            if (model.tensors.find(name.data()) == model.tensors.end()) {
                fprintf(stderr, "%s: unknown tensor '%s' in model file\n", __func__, name.data());
                return false;
            }

            auto tensor = model.tensors[name.data()];
            if (ggml_nelements(tensor) != nelements) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file\n", __func__, name.data());
                return false;
            }

            if (tensor->ne[0] != ne[0] || tensor->ne[1] != ne[1]) {
                fprintf(stderr, "%s: tensor '%s' has wrong shape in model file: got [%d, %d], expected [%d, %d]\n",
                        __func__, name.data(), (int) tensor->ne[0], (int) tensor->ne[1], ne[0], ne[1]);
                return false;
            }

            // for debugging
            if (0) {
                printf("%24s - [%5d, %5d], type = %6s, %6.2f MB, %9zu bytes\n", name.data(), ne[0], ne[1],
                       ggml_type_name(ggml_type(ttype)), ggml_nbytes(tensor) / 1024.0 / 1024.0, ggml_nbytes(tensor));
            }

            const size_t bpe = ggml_type_size(ggml_type(ttype));

            if ((nelements * bpe) / ggml_blck_size(tensor->type) != ggml_nbytes(tensor)) {
                fprintf(stderr, "%s: tensor '%s' has wrong size in model file: got %zu, expected %zu\n",
                        __func__, name.data(), ggml_nbytes(tensor), nelements * bpe);
                return false;
            }

            fin.read(reinterpret_cast<char *>(tensor->data), ggml_nbytes(tensor));

            //printf("%42s - [%5d, %5d], type = %6s, %6.2f MB\n", name.data(), ne[0], ne[1], ttype == 0 ? "float" : "f16", ggml_nbytes(tensor)/1024.0/1024.0);
            total_size += ggml_nbytes(tensor);
            if (++n_tensors % 8 == 0) {
                printf(".");
                fflush(stdout);
            }
        }

        printf(" done\n");

        printf("%s: model size = %8.2f MB / num tensors = %d\n", __func__, total_size / 1024.0 / 1024.0, n_tensors);
    }

    fin.close();

    return true;
}

// evaluate the transformer
//
//   - model:     the model
//   - n_threads: number of threads to use
//   - n_past:    the context size so far
//   - embd_inp:  the embeddings of the tokens in the context
//   - embd_w:    the predicted logits for the next token
//
// The GPT-J model requires about 16MB of memory per input token.
//
bool
gptj_eval(const gptj_model &model, const kv_cache &kv_cache, const int n_threads, const int n_past,
          const std::vector<gpt_vocab::id> &embd_inp,
          std::vector<float> &embd_w, size_t &mem_per_token, gptj_buffer &eval_buf) {
    const int N = embd_inp.size();

    const auto &hparams = model.hparams;

    const int n_embd = hparams.n_embd;
    const int n_layer = hparams.n_layer;
    const int n_ctx = hparams.n_ctx;
    const int n_head = hparams.n_head;
    const int n_vocab = hparams.n_vocab;
    const int n_rot = hparams.n_rot;

    struct ggml_init_params params = {
            .mem_size   = eval_buf.size,
            .mem_buffer = eval_buf.addr,
            .no_alloc   = false,
    };

    struct ggml_context *ctx0 = ggml_init(params);
    struct ggml_cgraph gf = {};
    gf.n_threads = n_threads;

    struct ggml_tensor *embd = ggml_new_tensor_1d(ctx0, GGML_TYPE_I32, N);
    memcpy(embd->data, embd_inp.data(), N * ggml_element_size(embd));

    // wte
    struct ggml_tensor *inpL = ggml_get_rows(ctx0, model.wte, embd);

    for (int il = 0; il < n_layer; ++il) {
        struct ggml_tensor *cur;

        // norm
        {
            cur = ggml_norm(ctx0, inpL);

            // cur = ln_1_g*cur + ln_1_b
            cur = ggml_add(ctx0,
                           ggml_mul(ctx0,
                                    ggml_repeat(ctx0, model.layers[il].ln_1_g, cur),
                                    cur),
                           ggml_repeat(ctx0, model.layers[il].ln_1_b, cur));
        }

        struct ggml_tensor *inpSA = cur;

        // self-attention
        {
            struct ggml_tensor *Qcur = ggml_rope(ctx0, ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0,
                                                                                          model.layers[il].c_attn_q_proj_w,
                                                                                          cur), n_embd / n_head, n_head,
                                                                       N), n_past, n_rot, 0);
            struct ggml_tensor *Kcur = ggml_rope(ctx0, ggml_reshape_3d(ctx0, ggml_mul_mat(ctx0,
                                                                                          model.layers[il].c_attn_k_proj_w,
                                                                                          cur), n_embd / n_head, n_head,
                                                                       N), n_past, n_rot, 0);

            // store key and value to memory
            {
                struct ggml_tensor *Vcur = ggml_transpose(ctx0,
                                                          ggml_mul_mat(ctx0, model.layers[il].c_attn_v_proj_w, cur));

                struct ggml_tensor *k = ggml_view_1d(ctx0, kv_cache.k, N * n_embd,
                                                     (ggml_element_size(kv_cache.k) * n_embd) *
                                                     (il * n_ctx + n_past));
                struct ggml_tensor *v = ggml_view_2d(ctx0, kv_cache.v, N, n_embd,
                                                     (n_ctx) * ggml_element_size(kv_cache.v),
                                                     (il * n_ctx) * ggml_element_size(kv_cache.v) * n_embd +
                                                     n_past * ggml_element_size(kv_cache.v));

                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Kcur, k));
                ggml_build_forward_expand(&gf, ggml_cpy(ctx0, Vcur, v));
            }

            // Q = Qcur.contiguous().view(n_embd/n_head, n_head, N).permute(0, 2, 1, 3)
            struct ggml_tensor *Q =
                    ggml_permute(ctx0,
                                 Qcur,
                                 0, 2, 1, 3);

            // K = Kmem.view(n_embd/n_head, n_head, n_past + N).permute(0, 2, 1, 3)
            struct ggml_tensor *K =
                    ggml_permute(ctx0,
                                 ggml_reshape_3d(ctx0,
                                                 ggml_view_1d(ctx0, kv_cache.k, (n_past + N) * n_embd,
                                                              il * n_ctx * ggml_element_size(kv_cache.k) * n_embd),
                                                 n_embd / n_head, n_head, n_past + N),
                                 0, 2, 1, 3);

            // K * Q
            struct ggml_tensor *KQ = ggml_mul_mat(ctx0, K, Q);

            // KQ_scaled = KQ / sqrt(n_embd/n_head)
            struct ggml_tensor *KQ_scaled =
                    ggml_scale(ctx0,
                               KQ,
                               ggml_new_f32(ctx0, 1.0f / sqrt(float(n_embd) / n_head))
                    );

            // KQ_masked = mask_past(KQ_scaled)
            struct ggml_tensor *KQ_masked = ggml_diag_mask_inf(ctx0, KQ_scaled, n_past);

            // KQ = soft_max(KQ_masked)
            struct ggml_tensor *KQ_soft_max = ggml_soft_max(ctx0, KQ_masked);

            // V_trans = Vmem.view(n_embd/n_head, n_head, n_past + N).permute(1, 2, 0, 3).contiguous()
            struct ggml_tensor *V =
                    ggml_view_3d(ctx0, kv_cache.v,
                                 n_past + N, n_embd / n_head, n_head,
                                 n_ctx * ggml_element_size(kv_cache.v),
                                 n_ctx * ggml_element_size(kv_cache.v) * n_embd / n_head,
                                 il * n_ctx * ggml_element_size(kv_cache.v) * n_embd);

            // KQV = transpose(V) * KQ_soft_max
            struct ggml_tensor *KQV = ggml_mul_mat(ctx0, V, KQ_soft_max);

            // KQV_merged = KQV.permute(0, 2, 1, 3)
            struct ggml_tensor *KQV_merged = ggml_permute(ctx0, KQV, 0, 2, 1, 3);

            // cur = KQV_merged.contiguous().view(n_embd, N)
            cur = ggml_cpy(ctx0,
                           KQV_merged,
                           ggml_new_tensor_2d(ctx0, GGML_TYPE_F32, n_embd, N));

            // projection (no bias)
            cur = ggml_mul_mat(ctx0,
                               model.layers[il].c_attn_proj_w,
                               cur);
        }

        struct ggml_tensor *inpFF = cur;

        // feed-forward network
        // this is independent of the self-attention result, so it could be done in parallel to the self-attention
        {
            // note here we pass inpSA instead of cur
            cur = ggml_mul_mat(ctx0,
                               model.layers[il].c_mlp_fc_w,
                               inpSA);

            cur = ggml_add(ctx0,
                           ggml_repeat(ctx0, model.layers[il].c_mlp_fc_b, cur),
                           cur);

            // GELU activation
            cur = ggml_gelu(ctx0, cur);

            // projection
            // cur = proj_w*cur + proj_b
            cur = ggml_mul_mat(ctx0,
                               model.layers[il].c_mlp_proj_w,
                               cur);

            cur = ggml_add(ctx0,
                           ggml_repeat(ctx0, model.layers[il].c_mlp_proj_b, cur),
                           cur);
        }

        // self-attention + FF
        cur = ggml_add(ctx0, cur, inpFF);

        // input for next layer
        inpL = ggml_add(ctx0, cur, inpL);
    }

    // norm
    {
        inpL = ggml_norm(ctx0, inpL);

        // inpL = ln_f_g*inpL + ln_f_b
        inpL = ggml_add(ctx0,
                        ggml_mul(ctx0,
                                 ggml_repeat(ctx0, model.ln_f_g, inpL),
                                 inpL),
                        ggml_repeat(ctx0, model.ln_f_b, inpL));
    }

    // lm_head
    {
        inpL = ggml_mul_mat(ctx0, model.lmh_g, inpL);

        inpL = ggml_add(ctx0,
                        ggml_repeat(ctx0, model.lmh_b, inpL),
                        inpL);
    }

    // logits -> probs
    //inpL = ggml_soft_max(ctx0, inpL);

    // run the computation
    ggml_build_forward_expand(&gf, inpL);
    ggml_graph_compute(ctx0, &gf);

    //if (n_past%100 == 0) {
    //    ggml_graph_print   (&gf);
    //    ggml_graph_dump_dot(&gf, NULL, "gpt-2.dot");
    //}

    //embd_w.resize(n_vocab*N);
    //memcpy(embd_w.data(), ggml_get_data(inpL), sizeof(float)*n_vocab*N);

    // return result for just the last token
    embd_w.resize(n_vocab);
    memcpy(embd_w.data(), (float *) ggml_get_data(inpL) + (n_vocab * (N - 1)), sizeof(float) * n_vocab);

    if (mem_per_token == 0) {
        mem_per_token = ggml_used_mem(ctx0) / N;
    }
    //printf("used_mem = %zu\n", ggml_used_mem(ctx0));

    ggml_free(ctx0);

    return true;
}

int main(int argc, char **argv) {
    const int64_t t_main_start_us = ggml_time_us();

    gpt_params params;
    params.model = "models/gpt-j-6B/ggml-model.bin";

    if (gpt_params_parse(argc, argv, params) == false) {
        return 1;
    }

    if (params.seed < 0) {
        params.seed = time(NULL);
    }

    printf("%s: seed = %d\n", __func__, params.seed);

    std::mt19937 rng(params.seed);
    if (params.prompt.empty()) {
        if (!isatty(STDIN_FILENO)) {
            std::string line;
            while (std::getline(std::cin, line)) {
                params.prompt = params.prompt + "\n" + line;
            }
        } else {
            params.prompt = gpt_random_prompt(rng);
        }
    }

    int64_t t_load_us = 0;

    gpt_vocab vocab;
    gptj_model model;
    kv_cache kv_cache;
    gptj_mlock mlock_mmap;

    // load the model
    {
        const int64_t t_start_us = ggml_time_us();

        if (!gptj_model_load_mmap(params.model, model, vocab, mlock_mmap)) {
            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
            return 1;
        }

        if (!kv_cache_init(model.hparams, kv_cache)) {
            fprintf(stderr, "%s: failed to load kv_cache from '%s'\n", __func__, params.model.c_str());
            return 1;
        }


//        if (!gptj_model_load(params.model, model, vocab)) {
//            fprintf(stderr, "%s: failed to load model from '%s'\n", __func__, params.model.c_str());
//            return 1;
//        }

        t_load_us = ggml_time_us() - t_start_us;
    }

    int n_past = 0;

    int64_t t_sample_us = 0;
    int64_t t_predict_us = 0;

    std::vector<float> logits;

    // tokenize the prompt
    std::vector<gpt_vocab::id> embd_inp = ::gpt_tokenize(vocab, params.prompt);

    params.n_predict = std::min(params.n_predict, model.hparams.n_ctx - (uint32_t) embd_inp.size());

    printf("%s: number of tokens in prompt = %zu\n", __func__, embd_inp.size());
    printf("\n");

    std::vector<gpt_vocab::id> embd;

    gptj_buffer eval_buf;
    eval_buf.resize(256ull * 1024 * 1024);

    // determine the required inference memory per token:
    size_t mem_per_token = 0;
    gptj_eval(model, kv_cache, params.n_threads, 0, {0, 1, 2, 3}, logits, mem_per_token, eval_buf);

    for (int i = embd.size(); i < embd_inp.size() + params.n_predict; i++) {
        // predict
        if (embd.size() > 0) {
            const int64_t t_start_us = ggml_time_us();

            if (!gptj_eval(model, kv_cache, params.n_threads, n_past, embd, logits, mem_per_token, eval_buf)) {
                printf("Failed to predict\n");
                return 1;
            }

            t_predict_us += ggml_time_us() - t_start_us;
        }

        n_past += embd.size();
        embd.clear();

        if (i >= embd_inp.size()) {
            // sample next token
            const int top_k = params.top_k;
            const float top_p = params.top_p;
            const float temp = params.temp;

            const int n_vocab = model.hparams.n_vocab;

            gpt_vocab::id id = 0;

            {
                const int64_t t_start_sample_us = ggml_time_us();

                id = gpt_sample_top_k_top_p(vocab, logits.data() + (logits.size() - n_vocab), top_k, top_p, temp, rng);

                t_sample_us += ggml_time_us() - t_start_sample_us;
            }

            // add it to the context
            embd.push_back(id);
        } else {
            // if here, it means we are still processing the input prompt
            for (int k = i; k < embd_inp.size(); k++) {
                embd.push_back(embd_inp[k]);
                if (embd.size() > params.n_batch) {
                    break;
                }
            }
            i += embd.size() - 1;
        }

        // display text
        for (auto id: embd) {
//            printf("%s: id=%d\n", __func__, id);
            printf("%s", vocab.id_to_token[id].c_str());
        }
        fflush(stdout);

        // end of text token
        if (embd.back() == 50256) {
            break;
        }
    }

    // report timing
    {
        const int64_t t_main_end_us = ggml_time_us();

        printf("\n\n");
        printf("%s: mem per token = %8zu bytes\n", __func__, mem_per_token);
        printf("%s:     load time = %8.2f ms\n", __func__, t_load_us / 1000.0f);
        printf("%s:   sample time = %8.2f ms\n", __func__, t_sample_us / 1000.0f);
        printf("%s:  predict time = %8.2f ms / %.2f ms per token\n", __func__, t_predict_us / 1000.0f,
               t_predict_us / 1000.0f / n_past);
        printf("%s:    total time = %8.2f ms\n", __func__, (t_main_end_us - t_main_start_us) / 1000.0f);
    }

    ggml_free(model.ctx);

    return 0;
}
