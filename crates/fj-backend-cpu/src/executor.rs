//! CPU execution engine wrapping eval_jaxpr.
//!
//! The CpuBackend provides the Backend trait implementation for host-CPU
//! execution. All computation runs via fj-interpreters::eval_jaxpr.
//!
//! Contract: p2c006.strict.inv001 (CPU always available).

use fj_core::{Jaxpr, Value};
use fj_runtime::backend::{Backend, BackendError};
use fj_runtime::buffer::Buffer;
use fj_runtime::device::{DeviceId, DeviceInfo, Platform};

/// CPU backend: interprets Jaxpr programs on the host CPU.
///
/// V1 scope: single CPU device (DeviceId(0)). All computation is
/// synchronous and single-threaded.
pub struct CpuBackend {
    /// Number of logical CPU devices to expose.
    /// V1: always 1.
    device_count: u32,
    /// Version string for cache key inclusion.
    version_string: String,
}

impl CpuBackend {
    /// Create a CPU backend with a single device.
    #[must_use]
    pub fn new() -> Self {
        Self {
            device_count: 1,
            version_string: format!("fj-backend-cpu/{}", env!("CARGO_PKG_VERSION")),
        }
    }

    /// Create a CPU backend exposing multiple logical devices.
    /// Useful for testing multi-device dispatch without GPU hardware.
    #[must_use]
    pub fn with_device_count(count: u32) -> Self {
        assert!(count > 0, "device count must be at least 1");
        Self {
            device_count: count,
            version_string: format!("fj-backend-cpu/{}", env!("CARGO_PKG_VERSION")),
        }
    }
}

impl Default for CpuBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl Backend for CpuBackend {
    fn name(&self) -> &str {
        "cpu"
    }

    fn devices(&self) -> Vec<DeviceInfo> {
        (0..self.device_count)
            .map(|i| DeviceInfo {
                id: DeviceId(i),
                platform: Platform::Cpu,
                host_id: 0,
                process_index: 0,
            })
            .collect()
    }

    fn default_device(&self) -> DeviceId {
        DeviceId(0)
    }

    fn execute(
        &self,
        jaxpr: &Jaxpr,
        args: &[Value],
        _device: DeviceId,
    ) -> Result<Vec<Value>, BackendError> {
        // CPU backend ignores device ID — all execution is on the host.
        fj_interpreters::eval_jaxpr(jaxpr, args)
            .map_err(|e| BackendError::ExecutionFailed {
                detail: e.to_string(),
            })
    }

    fn allocate(&self, size_bytes: usize, device: DeviceId) -> Result<Buffer, BackendError> {
        if device.0 >= self.device_count {
            return Err(BackendError::AllocationFailed {
                device,
                detail: format!("device {} not available (have {})", device.0, self.device_count),
            });
        }
        Ok(Buffer::zeroed(size_bytes, device))
    }

    fn transfer(
        &self,
        buffer: &Buffer,
        target: DeviceId,
    ) -> Result<Buffer, BackendError> {
        if target.0 >= self.device_count {
            return Err(BackendError::TransferFailed {
                source: buffer.device(),
                target,
                detail: format!("target device {} not available", target.0),
            });
        }
        // CPU "transfer" is a clone (same memory space).
        Ok(Buffer::new(buffer.as_bytes().to_vec(), target))
    }

    fn version(&self) -> &str {
        &self.version_string
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use fj_core::{ProgramSpec, build_program};

    #[test]
    fn cpu_backend_name() {
        let backend = CpuBackend::new();
        assert_eq!(backend.name(), "cpu");
    }

    #[test]
    fn cpu_backend_default_device() {
        let backend = CpuBackend::new();
        assert_eq!(backend.default_device(), DeviceId(0));
    }

    #[test]
    fn cpu_backend_single_device_discovery() {
        let backend = CpuBackend::new();
        let devices = backend.devices();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].id, DeviceId(0));
        assert_eq!(devices[0].platform, Platform::Cpu);
        assert_eq!(devices[0].host_id, 0);
        assert_eq!(devices[0].process_index, 0);
    }

    #[test]
    fn cpu_backend_multi_device_discovery() {
        let backend = CpuBackend::with_device_count(4);
        let devices = backend.devices();
        assert_eq!(devices.len(), 4);
        for (i, dev) in devices.iter().enumerate() {
            assert_eq!(dev.id, DeviceId(i as u32));
            assert_eq!(dev.platform, Platform::Cpu);
        }
    }

    #[test]
    fn cpu_backend_execute_add2() {
        let backend = CpuBackend::new();
        let jaxpr = build_program(ProgramSpec::Add2);
        let result = backend
            .execute(
                &jaxpr,
                &[Value::scalar_i64(3), Value::scalar_i64(4)],
                DeviceId(0),
            )
            .expect("execution should succeed");
        assert_eq!(result, vec![Value::scalar_i64(7)]);
    }

    #[test]
    fn cpu_backend_execute_square() {
        let backend = CpuBackend::new();
        let jaxpr = build_program(ProgramSpec::Square);
        let result = backend
            .execute(&jaxpr, &[Value::scalar_f64(5.0)], DeviceId(0))
            .expect("execution should succeed");
        let val = result[0].as_f64_scalar().expect("should be f64");
        assert!((val - 25.0).abs() < 1e-10);
    }

    #[test]
    fn cpu_backend_allocate_and_access() {
        let backend = CpuBackend::new();
        let buf = backend.allocate(256, DeviceId(0)).expect("alloc should succeed");
        assert_eq!(buf.size(), 256);
        assert_eq!(buf.device(), DeviceId(0));
        assert!(buf.as_bytes().iter().all(|&b| b == 0));
    }

    #[test]
    fn cpu_backend_allocate_invalid_device() {
        let backend = CpuBackend::new();
        let err = backend.allocate(256, DeviceId(1)).expect_err("should fail");
        assert!(matches!(err, BackendError::AllocationFailed { .. }));
    }

    #[test]
    fn cpu_backend_transfer_same_device() {
        let backend = CpuBackend::new();
        let buf = Buffer::new(vec![1, 2, 3], DeviceId(0));
        let transferred = backend.transfer(&buf, DeviceId(0)).expect("transfer should succeed");
        assert_eq!(transferred.as_bytes(), &[1, 2, 3]);
        assert_eq!(transferred.device(), DeviceId(0));
    }

    #[test]
    fn cpu_backend_transfer_cross_device() {
        let backend = CpuBackend::with_device_count(2);
        let buf = Buffer::new(vec![10, 20, 30], DeviceId(0));
        let transferred = backend.transfer(&buf, DeviceId(1)).expect("cross-device transfer");
        assert_eq!(transferred.as_bytes(), &[10, 20, 30]);
        assert_eq!(transferred.device(), DeviceId(1));
        // Original buffer unchanged
        assert_eq!(buf.device(), DeviceId(0));
    }

    #[test]
    fn cpu_backend_transfer_invalid_target() {
        let backend = CpuBackend::new();
        let buf = Buffer::new(vec![1], DeviceId(0));
        let err = backend.transfer(&buf, DeviceId(5)).expect_err("should fail");
        assert!(matches!(err, BackendError::TransferFailed { .. }));
    }

    #[test]
    fn cpu_backend_version_string() {
        let backend = CpuBackend::new();
        assert!(backend.version().starts_with("fj-backend-cpu/"));
    }

    #[test]
    fn cpu_backend_buffer_roundtrip_preserves_data() {
        // Contract p2c006.strict.inv003: device_put/device_get round-trip
        let backend = CpuBackend::new();
        let original = vec![0xCA, 0xFE, 0xBA, 0xBE];
        let buf = Buffer::new(original.clone(), DeviceId(0));
        let data = buf.into_bytes();
        assert_eq!(original, data);

        // Through allocate + write
        let mut buf = backend.allocate(4, DeviceId(0)).expect("alloc");
        buf.as_bytes_mut().copy_from_slice(&original);
        assert_eq!(buf.as_bytes(), &original[..]);
    }
}
