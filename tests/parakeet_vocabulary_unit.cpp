// Generic vocabulary dispatch and ownership against tiny Parakeet fixtures.
// The fixtures exercise loading/stream initialization, not recognition quality.
// Internal inspection checks that accepted generic vocabulary actually reaches
// the decoder, rather than merely being ignored by the public dispatcher.

#include "arch/parakeet/parakeet.h"
#include "transcribe.h"
#include "transcribe/parakeet.h"

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <string>
#include <vector>

namespace {

int failures = 0;

#define CHECK(condition)                                                              \
    do {                                                                              \
        if (!(condition)) {                                                           \
            std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
            ++failures;                                                               \
        }                                                                             \
    } while (0)

struct Fixture {
    transcribe_model *   model   = nullptr;
    transcribe_session * session = nullptr;

    ~Fixture() {
        transcribe_session_free(session);
        transcribe_model_free(model);
    }

    bool load(const char * name) {
        transcribe_model_load_params params;
        transcribe_model_load_params_init(&params);
        params.backend         = TRANSCRIBE_BACKEND_CPU;
        const std::string path = std::string(TRANSCRIBE_TEST_FIXTURES_DIR) + "/" + name;
        const auto        st   = transcribe_model_load_file(path.c_str(), &params, &model);
        CHECK(st == TRANSCRIBE_OK);
        if (st != TRANSCRIBE_OK) {
            return false;
        }
        const auto init = transcribe_session_init(model, nullptr, &session);
        CHECK(init == TRANSCRIBE_OK);
        return init == TRANSCRIBE_OK;
    }

    transcribe::parakeet::ParakeetSession * state() {
        return static_cast<transcribe::parakeet::ParakeetSession *>(session);
    }

    transcribe_status begin(transcribe_run_params & rp, const transcribe_ext * extension = nullptr) {
        transcribe_stream_reset(session);
        transcribe_stream_params sp;
        transcribe_stream_params_init(&sp);
        sp.family = extension;
        return transcribe_stream_begin(session, &rp, &sp);
    }
};

void test_generic_vocabulary_and_ownership(const char * fixture_name) {
    Fixture f;
    if (!f.load(fixture_name)) {
        return;
    }
    CHECK(transcribe_model_supports(f.model, TRANSCRIBE_FEATURE_VOCABULARY));

    transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    CHECK(f.begin(rp) == TRANSCRIBE_OK);
    CHECK(!f.state()->stream_bias.has_value());

    std::string  term    = "hello";
    const char * terms[] = { term.c_str(), "hello", "", "   " };
    rp.vocabulary        = terms;
    rp.n_vocabulary      = 4;
    CHECK(f.begin(rp) == TRANSCRIBE_OK);
    CHECK(f.state()->stream_bias.has_value());
    if (f.state()->stream_bias) {
        CHECK(f.state()->stream_bias->boost() == 3.0f);
        CHECK(!f.state()->stream_bias->empty());
    }
    // Stream initialization must own the vocabulary and compile the decoder
    // state before the caller mutates/releases its strings and pointer array.
    term.assign(2048, 'x');
    terms[0] = nullptr;
    CHECK(f.state()->stream_run_params.n_vocabulary > 0);
    CHECK(std::strcmp(f.state()->stream_run_params.vocabulary[0], "hello") == 0);
    CHECK(f.state()->stream_dec_state.bias_state == 0);

    // A subsequent empty vocabulary cannot retain the previous automaton.
    rp.vocabulary   = nullptr;
    rp.n_vocabulary = 0;
    CHECK(f.begin(rp) == TRANSCRIBE_OK);
    CHECK(!f.state()->stream_bias.has_value());
}

// Layouts of the upstream v1 extensions. Poison their trailing padding so a
// new scalar placed inside that padding cannot accidentally be read as boost.
struct CacheV1 {
    transcribe_ext ext;
    int32_t        att_context_right;
};

struct BufferedV1 {
    transcribe_ext ext;
    int32_t        left_ms;
    int32_t        chunk_ms;
    int32_t        right_ms;
};

static_assert(offsetof(transcribe_parakeet_stream_ext, bias_boost) >= sizeof(CacheV1));
static_assert(offsetof(transcribe_parakeet_buffered_stream_ext, bias_boost) >= sizeof(BufferedV1));

void test_boost_and_old_layout(bool buffered) {
    Fixture f;
    if (!f.load(buffered ? "tokenizer_minimal_streaming_buffered.gguf" :
                           "tokenizer_minimal_streaming_cache_aware.gguf")) {
        return;
    }
    const char *          terms[] = { "hello" };
    transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    rp.vocabulary   = terms;
    rp.n_vocabulary = 1;

    CacheV1 old_cache;
    std::memset(&old_cache, 0xFF, sizeof(old_cache));
    old_cache.ext.size          = sizeof(old_cache);
    old_cache.ext.kind          = TRANSCRIBE_EXT_KIND_PARAKEET_STREAM;
    old_cache.att_context_right = -1;
    BufferedV1 old_buffered;
    std::memset(&old_buffered, 0xFF, sizeof(old_buffered));
    old_buffered.ext.size = sizeof(old_buffered);
    old_buffered.ext.kind = TRANSCRIBE_EXT_KIND_PARAKEET_BUFFERED_STREAM;
    old_buffered.left_ms = old_buffered.chunk_ms = old_buffered.right_ms = -1;
    CHECK(f.begin(rp, buffered ? &old_buffered.ext : &old_cache.ext) == TRANSCRIBE_OK);
    CHECK(f.state()->stream_bias.has_value());
    if (f.state()->stream_bias) {
        CHECK(f.state()->stream_bias->boost() == 3.0f);
    }

    transcribe_parakeet_stream_ext cache;
    transcribe_parakeet_stream_ext_init(&cache);
    transcribe_parakeet_buffered_stream_ext buffer;
    transcribe_parakeet_buffered_stream_ext_init(&buffer);
    for (float boost : { 0.0f, 1.5f, 10.0f }) {
        cache.bias_boost = buffer.bias_boost = boost;
        CHECK(f.begin(rp, buffered ? &buffer.ext : &cache.ext) == TRANSCRIBE_OK);
        CHECK(f.state()->stream_bias.has_value() == (boost > 0.0f));
        if (f.state()->stream_bias) {
            CHECK(f.state()->stream_bias->boost() == boost);
        }
    }
    for (float boost :
         { -1.0f, 10.01f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN() }) {
        cache.bias_boost = buffer.bias_boost = boost;
        CHECK(f.begin(rp, buffered ? &buffer.ext : &cache.ext) == TRANSCRIBE_ERR_INVALID_ARG);
    }
}

void test_invalid_vocabulary(bool streaming) {
    Fixture f;
    if (!f.load(streaming ? "tokenizer_minimal_streaming_cache_aware.gguf" : "tokenizer_minimal.gguf")) {
        return;
    }
    CHECK(transcribe_model_supports(f.model, TRANSCRIBE_FEATURE_VOCABULARY));
    transcribe_run_params rp;
    transcribe_run_params_init(&rp);
    const float pcm[]  = { 0.0f };
    auto        reject = [&]() {
        const auto st = streaming ? f.begin(rp) : transcribe_run(f.session, pcm, 1, &rp);
        CHECK(st == TRANSCRIBE_ERR_INVALID_ARG);
    };

    rp.n_vocabulary = 1;  // Nonzero count requires an array.
    reject();
    const char * missing[] = { nullptr };
    rp.vocabulary          = missing;
    reject();
    std::vector<const char *> too_many(257, "hello");
    rp.vocabulary   = too_many.data();
    rp.n_vocabulary = static_cast<int32_t>(too_many.size());
    reject();
    const std::string too_long(257, 'a');
    const char *      long_term[] = { too_long.c_str() };
    rp.vocabulary                 = long_term;
    rp.n_vocabulary               = 1;
    reject();

    for (const std::string invalid : { std::string("\xC0\xAF"), std::string("\xED\xA0\x80"),
                                       std::string("\xF4\x90\x80\x80"), std::string("\xC3") }) {
        const char * invalid_term[] = { invalid.c_str() };
        rp.vocabulary               = invalid_term;
        reject();
    }
    if (streaming) {
        std::vector<const char *> at_limit(256, "hello");
        rp.vocabulary   = at_limit.data();
        rp.n_vocabulary = 256;
        CHECK(f.begin(rp) == TRANSCRIBE_OK);
        CHECK(f.state()->stream_bias.has_value());
        rp.n_vocabulary = 1;
    }

    if (!streaming) {
        const char * term[] = { "hello" };
        rp.vocabulary       = term;
        transcribe_parakeet_bias_run_ext ext;
        transcribe_parakeet_bias_run_ext_init(&ext);
        rp.family = &ext.ext;
        for (float boost :
             { -1.0f, 10.01f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN() }) {
            ext.boost = boost;
            reject();
        }
    }
}

}  // namespace

int main() {
    test_generic_vocabulary_and_ownership("tokenizer_minimal_streaming_cache_aware.gguf");
    test_generic_vocabulary_and_ownership("tokenizer_minimal_streaming_buffered.gguf");
    test_boost_and_old_layout(false);
    test_boost_and_old_layout(true);
    test_invalid_vocabulary(false);
    test_invalid_vocabulary(true);
    std::printf("parakeet_vocabulary_unit: %d failures\n", failures);
    return failures == 0 ? 0 : 1;
}
