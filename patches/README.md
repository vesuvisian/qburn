# Crate patches

Wired from the workspace root via:

```toml
[patch.crates-io]
cubecl-runtime = { path = "patches/cubecl-runtime" }
```

After editing anything under `patches/`, rebuild wasm (`wasm-pack build web …`) and re-stage.

## `cubecl-runtime` (Burn 0.23 / CubeCL 0.12-pre)

Vendored from crates.io **`cubecl-runtime 0.12.0-pre.2`**. One fix for wasm:

Upstream gates `tune::sampler` behind `not(wasm)`, but `tuner.rs` still calls
`measured` from that module when resolving autotune benches. That breaks
`wasm32-unknown-unknown` builds of Burn `webgpu`. This patch keeps `mod sampler`
available on wasm, with only `measured` compiled in (SampleSet / TimeChange stay
native-only with the adaptive schedule).

With Burn **0.23.0-pre.2**, matmul goes through cubek’s **Hardware-aware** tile-DSL
selector (`MatmulKernelHeuristic::Tile`): kernels are planned against the
device’s `max_shared_memory_size`.

Autotune stays **off** for the web crate. Do not reintroduce oversized autotune
candidates on WebGPU (invalid WGSL can still poison the device). The `qburn`
`webgpu` feature also omits `burn/fusion` (Burn documents that fusion may need
to be disabled on wasm).

See also [web/static/architecture.html](../web/static/architecture.html#webgpu-matmul).
