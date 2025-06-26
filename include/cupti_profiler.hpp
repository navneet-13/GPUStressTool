#pragma once

// Initializes CUPTI or stubs if not implemented yet
void init_cupti_profiler();
void finalize_cupti_profiler();
