#include "cupti_profiler.hpp"

#include <iostream>
#include <cupti.h>

#define CHECK_CUPTI(call)                                              \
    do {                                                               \
        CUptiResult _status = call;                                    \
        if (_status != CUPTI_SUCCESS) {                                \
            const char* errstr;                                        \
            cuptiGetResultString(_status, &errstr);                    \
            std::cerr << "CUPTI error: " << errstr << std::endl;       \
            exit(EXIT_FAILURE);                                        \
        }                                                              \
    } while (0)

#define ACTIVITY_BUFFER_SIZE (32 * 1024)

void CUPTIAPI bufferRequested(uint8_t **buffer, size_t *size, size_t *maxNumRecords) {
    *size = ACTIVITY_BUFFER_SIZE;
    *buffer = (uint8_t *)malloc(*size);
    *maxNumRecords = 0; // let CUPTI decide
}

void CUPTIAPI bufferCompleted(CUcontext ctx, uint32_t streamId,
                              uint8_t *buffer, size_t size,
                              size_t validSize) {
    CUpti_Activity *record = nullptr;

    while (cuptiActivityGetNextRecord(buffer, validSize, &record) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_KERNEL) {
            auto *k = (CUpti_ActivityKernel3 *)record;
            std::cout << "[KERNEL] " << k->name << " ran on stream " << k->streamId
                      << " for " << k->end - k->start << " ns\n";
        } else if (record->kind == CUPTI_ACTIVITY_KIND_MEMCPY) {
            auto *m = (CUpti_ActivityMemcpy *)record;
            std::cout << "[MEMCPY] " << m->bytes << " bytes, from " << m->copyKind
                      << ", duration " << m->end - m->start << " ns\n";
        }
    }

    // Check for dropped records
    size_t dropped = 0;
    cuptiActivityGetNumDroppedRecords(ctx, streamId, &dropped);
    if (dropped > 0)
        std::cerr << "[CUPTI] Dropped " << dropped << " records\n";

    free(buffer);
}

void init_cupti_profiler() {
    std::cout << "[CUPTI] Enabling activity collection...\n";

    CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_KERNEL));
    CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY));
    CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET));
    CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER));
    CHECK_CUPTI(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME));
    CHECK_CUPTI(cuptiActivityRegisterCallbacks(bufferRequested, bufferCompleted));
}

void finalize_cupti_profiler() {
    std::cout << "[CUPTI] Flushing activity buffers...\n";
    CHECK_CUPTI(cuptiActivityFlushAll(0));
}