//! Resolve Burn device from crate features.

use burn::prelude::Device;

/// Pick a device from the enabled backend features.
///
/// On browser WebGPU this returns a **lazy** handle. Call
/// [`resolve_device_async`] (or `Device::wgpu_options().init_async()`) before
/// creating tensors — otherwise CubeCL panics with "Service not yet initialized".
pub fn resolve_device() -> Device {
    // Prefer native Metal / CUDA over wgpu when those features are on — wgpu on
    // Apple often advertises a 32 KiB workgroup limit and selects slower matmuls.
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    {
        return Device::metal();
    }
    #[cfg(feature = "cuda")]
    {
        return Device::cuda(0);
    }
    // Explicit WebGPU (WGSL) — browser wasm, or native `--features webgpu` probes.
    #[cfg(all(
        feature = "webgpu",
        target_family = "wasm",
        not(feature = "metal"),
        not(feature = "cuda")
    ))]
    {
        use burn::tensor::DeviceKind;
        return Device::webgpu(DeviceKind::DefaultDevice);
    }
    #[cfg(all(
        feature = "webgpu",
        not(target_family = "wasm"),
        not(feature = "metal"),
        not(feature = "cuda")
    ))]
    {
        use burn::tensor::wgpu::WgpuBackend;
        // Prefer true WebGPU/WGSL when the host supports it (closer to the
        // browser). macOS often rejects this API — fall back to Auto wgpu
        // still compiled *without* burn/fusion (the webgpu feature set).
        match Device::wgpu_options()
            .graphics_api(WgpuBackend::WebGpu)
            .init()
        {
            Ok(d) => return d,
            Err(e) => {
                eprintln!("webgpu graphics API unavailable ({e:?}); using Auto wgpu (fusion=off)");
                return Device::wgpu_options()
                    .init()
                    .unwrap_or_else(|_| Device::default());
            }
        }
    }
    #[cfg(all(
        any(feature = "wgpu", feature = "flex"),
        not(feature = "webgpu"),
        not(all(feature = "metal", target_vendor = "apple")),
        not(feature = "cuda")
    ))]
    {
        Device::default()
    }
    #[cfg(not(any(
        feature = "wgpu",
        feature = "flex",
        feature = "metal",
        feature = "cuda",
        feature = "webgpu"
    )))]
    compile_error!("Enable at least one of: wgpu, flex, metal, cuda, webgpu")
}

/// Short label for logs / `--metrics` (which backend `resolve_device` picked).
pub fn backend_name() -> &'static str {
    #[cfg(all(feature = "metal", target_vendor = "apple"))]
    {
        return "metal";
    }
    #[cfg(feature = "cuda")]
    {
        return "cuda";
    }
    #[cfg(all(feature = "webgpu", not(feature = "metal"), not(feature = "cuda")))]
    {
        return "webgpu";
    }
    #[cfg(all(feature = "wgpu-msl", not(feature = "metal"), not(feature = "cuda")))]
    {
        return "wgpu-msl";
    }
    #[cfg(all(
        feature = "wgpu",
        not(feature = "metal"),
        not(feature = "cuda"),
        not(feature = "webgpu")
    ))]
    {
        return "wgpu";
    }
    #[cfg(feature = "flex")]
    {
        return "flex";
    }
    #[allow(unreachable_code)]
    "unknown"
}

/// Initialize the compute runtime and return a ready device.
///
/// Required on wasm/WebGPU (and useful anywhere you want explicit wgpu options).
/// Native `wgpu` / `metal` / `cuda` paths that already work with [`resolve_device`]
/// still succeed here.
#[allow(clippy::unused_async)] // async only required for wasm wgpu init
pub async fn resolve_device_async() -> Result<Device, String> {
    #[cfg(all(feature = "webgpu", not(feature = "metal"), not(feature = "cuda")))]
    {
        use burn::tensor::wgpu::WgpuBackend;
        return Device::wgpu_options()
            .graphics_api(WgpuBackend::WebGpu)
            .init_async()
            .await
            .map_err(|e| format!("webgpu init: {e}"));
    }
    #[cfg(all(feature = "wgpu", not(feature = "webgpu"), not(target_family = "wasm")))]
    {
        // Explicit init keeps options configurable; falls back to sync default.
        match Device::wgpu_options().init() {
            Ok(d) => return Ok(d),
            Err(_) => return Ok(resolve_device()),
        }
    }
    #[cfg(not(any(
        all(feature = "webgpu", not(feature = "metal"), not(feature = "cuda")),
        all(feature = "wgpu", not(feature = "webgpu"), not(target_family = "wasm")),
    )))]
    {
        Ok(resolve_device())
    }
}
