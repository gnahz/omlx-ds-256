/* Read-only macOS telemetry. No Metal, allocations in the target, or controls. */
#import <Foundation/Foundation.h>
#include <errno.h>
#include <inttypes.h>
#include <libproc.h>
#include <mach/mach.h>
#include <stddef.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/sysctl.h>

struct writer { char *p; size_t left; int failed; };
static void put(struct writer *w, const char *fmt, ...) {
    if (w->failed) return;
    va_list ap; va_start(ap, fmt);
    int n = vsnprintf(w->p, w->left, fmt, ap); va_end(ap);
    if (n < 0 || (size_t)n >= w->left) { w->failed = 1; return; }
    w->p += n; w->left -= (size_t)n;
}

const char *r10_abi_json(void) {
    static char json[2048];
    snprintf(json, sizeof(json),
        "{\"abi_version\":1,\"integer_bytes\":%zu,\"rusage_v4_bytes\":%zu,"
        "\"rusage_phys_offset\":%zu,\"rusage_resident_offset\":%zu,"
        "\"rusage_disk_read_offset\":%zu,\"rusage_start_offset\":%zu,"
        "\"vm_struct_bytes\":%zu,\"host_vm_info64_count\":%u,"
        "\"vm_decompressions_offset\":%zu,\"vm_compressions_offset\":%zu,"
        "\"vm_swapins_offset\":%zu,\"vm_swapouts_offset\":%zu,"
        "\"vm_compressor_page_count_offset\":%zu,\"vm_uncompressed_pages_offset\":%zu,"
        "\"xsw_usage_bytes\":%zu}",
        sizeof(integer_t), sizeof(struct rusage_info_v4),
        offsetof(struct rusage_info_v4, ri_phys_footprint),
        offsetof(struct rusage_info_v4, ri_resident_size),
        offsetof(struct rusage_info_v4, ri_diskio_bytesread),
        offsetof(struct rusage_info_v4, ri_proc_start_abstime),
        sizeof(vm_statistics64_data_t), (unsigned)HOST_VM_INFO64_COUNT,
        offsetof(vm_statistics64_data_t, decompressions),
        offsetof(vm_statistics64_data_t, compressions),
        offsetof(vm_statistics64_data_t, swapins),
        offsetof(vm_statistics64_data_t, swapouts),
        offsetof(vm_statistics64_data_t, compressor_page_count),
        offsetof(vm_statistics64_data_t, total_uncompressed_pages_in_compressor),
        sizeof(struct xsw_usage));
    return json;
}

static void scalar(struct writer *w, const char *name) {
    uint64_t value = 0; size_t size = sizeof(value);
    errno = 0; int rc = sysctlbyname(name, &value, &size, NULL, 0); int error = errno;
    if (rc == 0 && (size == 4 || size == 8))
        put(w, "\"%s\":{\"available\":true,\"value\":%" PRIu64 ",\"width\":%zu}", name, value, size);
    else put(w, "\"%s\":{\"available\":false,\"errno\":%d,\"returned_bytes\":%zu}", name, error, size);
}

int r10_snapshot_json(int pid, char *output, size_t capacity) {
    if (!output || capacity < 2 || pid <= 0) return EINVAL;
    struct writer w = {output, capacity, 0};
    struct rusage_info_v4 ru = {0};
    errno = 0;
    int prc = proc_pid_rusage(pid, RUSAGE_INFO_V4, (rusage_info_t *)&ru);
    int perr = errno;
    put(&w, "{\"pid\":%d,\"process\":{\"available\":%s,\"rc\":%d,\"errno\":%d", pid, prc == 0 ? "true" : "false", prc, perr);
    if (prc == 0) {
#define PROC(name) put(&w, ",\"" #name "\":%" PRIu64, ru.ri_##name)
        PROC(user_time); PROC(system_time); PROC(pageins); PROC(wired_size);
        PROC(resident_size); PROC(phys_footprint); PROC(proc_start_abstime);
        PROC(lifetime_max_phys_footprint); PROC(diskio_bytesread); PROC(diskio_byteswritten);
#undef PROC
    }
    put(&w, "},\"host_vm\":{");
    vm_statistics64_data_t vm = {0};
    mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
    host_t host = mach_host_self(); vm_size_t page_size = 0;
    kern_return_t page_rc = host_page_size(host, &page_size);
    kern_return_t rc = host_statistics64(host, HOST_VM_INFO64, (host_info64_t)&vm, &count);
    mach_port_deallocate(mach_task_self(), host);
    put(&w, "\"available\":%s,\"rc\":%d,\"returned_words\":%u,\"requested_words\":%u,\"page_size_rc\":%d,\"page_bytes\":%" PRIu64,
        rc == KERN_SUCCESS ? "true" : "false", rc, count, (unsigned)HOST_VM_INFO64_COUNT, page_rc, (uint64_t)page_size);
    if (rc == KERN_SUCCESS) {
#define VM(name) do { \
    if ((size_t)count * sizeof(integer_t) >= offsetof(vm_statistics64_data_t, name) + sizeof(vm.name)) \
        put(&w, ",\"" #name "\":%" PRIu64, (uint64_t)vm.name); \
    else put(&w, ",\"" #name "\":null"); \
} while (0)
        VM(free_count); VM(active_count); VM(inactive_count); VM(wire_count);
        VM(pageins); VM(pageouts); VM(faults); VM(cow_faults); VM(purges);
        VM(purgeable_count); VM(speculative_count); VM(decompressions); VM(compressions);
        VM(swapins); VM(swapouts); VM(compressor_page_count); VM(throttled_count);
        VM(external_page_count); VM(internal_page_count);
        VM(total_uncompressed_pages_in_compressor); VM(swapped_count);
#undef VM
    }
    put(&w, "},\"sysctl\":{");
    const char *names[] = {
        "kern.memorystatus_vm_pressure_level", "kern.thermal_level",
        "machdep.xcpm.cpu_thermal_level", "machdep.xcpm.gpu_thermal_level",
        "vm.page_free_count", "vm.compressor.pages_compressed_incore",
        "vm.page_pageable_internal_count", "vm.page_pageable_external_count",
        "hw.memsize"
    };
    for (size_t i = 0; i < sizeof(names) / sizeof(names[0]); ++i) {
        if (i) put(&w, ","); scalar(&w, names[i]);
    }
    struct xsw_usage swap = {0}; size_t swap_size = sizeof(swap);
    errno = 0; int src = sysctlbyname("vm.swapusage", &swap, &swap_size, NULL, 0); int serr = errno;
    put(&w, "},\"swap_usage\":{\"available\":%s,\"errno\":%d", src == 0 && swap_size == sizeof(swap) ? "true" : "false", serr);
    if (src == 0 && swap_size == sizeof(swap))
        put(&w, ",\"total_bytes\":%" PRIu64 ",\"available_bytes\":%" PRIu64 ",\"used_bytes\":%" PRIu64 ",\"page_bytes\":%u,\"encrypted\":%s",
            swap.xsu_total, swap.xsu_avail, swap.xsu_used, swap.xsu_pagesize, swap.xsu_encrypted ? "true" : "false");
    put(&w, "},\"system_thermal\":{");
    @autoreleasepool {
        if (@available(macOS 10.10.3, *)) {
            NSProcessInfoThermalState state = [[NSProcessInfo processInfo] thermalState];
            const char *name = "unknown";
            switch (state) {
                case NSProcessInfoThermalStateNominal: name = "nominal"; break;
                case NSProcessInfoThermalStateFair: name = "fair"; break;
                case NSProcessInfoThermalStateSerious: name = "serious"; break;
                case NSProcessInfoThermalStateCritical: name = "critical"; break;
            }
            put(&w, "\"available\":true,\"provider\":\"NSProcessInfo.thermalState\",\"raw\":%ld,\"state\":\"%s\",\"scope\":\"system thermal state observed by collector\"", (long)state, name);
        } else {
            put(&w, "\"available\":false,\"provider\":\"NSProcessInfo.thermalState\",\"reason\":\"OS API unavailable\"");
        }
    }
    put(&w, "}}");
    return w.failed ? ENOSPC : 0;
}
