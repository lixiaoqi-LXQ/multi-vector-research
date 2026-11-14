#pragma once
#include <cstdint>
#include <nvtx3/nvToolsExt.h>
namespace Chimera {
// Host scopes annotate submission, waits and CPU execution. They add no CUDA
// synchronization and are not GPU-exclusive durations. Disabled by default.
class ProfileRange {
 public:
    ProfileRange(bool enabled, const char* name, int64_t payload = -1) : enabled_(enabled) {
        if (!enabled_) return;
        nvtxEventAttributes_t attr{};
        attr.version = NVTX_VERSION;
        attr.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
        attr.messageType = NVTX_MESSAGE_TYPE_ASCII;
        attr.message.ascii = name;
        if (payload >= 0) {
            attr.payloadType = NVTX_PAYLOAD_TYPE_INT64;
            attr.payload.llValue = payload;
        }
        nvtxRangePushEx(&attr);
    }
    void close() noexcept { if (enabled_) { nvtxRangePop(); enabled_ = false; } }
    ~ProfileRange() { close(); }
    ProfileRange(const ProfileRange&) = delete;
    ProfileRange& operator=(const ProfileRange&) = delete;
 private:
    bool enabled_;
};
}
