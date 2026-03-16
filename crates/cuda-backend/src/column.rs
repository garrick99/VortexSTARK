//! GPU-resident column type for stwo's Column<T> trait.
//!
//! Wraps VortexSTARK's DeviceBuffer with the interface stwo expects.
//! Data lives on GPU; `to_cpu()` downloads on demand.

use stwo::prover::backend::{Column, ColumnOps};
use stwo::core::fields::m31::BaseField;
use stwo::core::fields::qm31::SecureField;
use vortexstark::cuda::ffi;
use vortexstark::device::DeviceBuffer;
use std::ffi::c_void;

use super::CudaBackend;

/// GPU-resident column. Stores `len` field elements on the device.
/// The inner DeviceBuffer stores raw u32s (M31 values are u32, QM31 is 4×u32).
#[derive(Clone, Debug)]
pub struct CudaColumn<T> {
    /// Device memory. For BaseField: 1 u32 per element. For SecureField: 4 u32s per element.
    pub(crate) buf: DeviceBuffer<u32>,
    pub(crate) len: usize,
    _marker: std::marker::PhantomData<T>,
}

impl<T> CudaColumn<T> {
    pub fn from_device_buffer(buf: DeviceBuffer<u32>, len: usize) -> Self {
        Self { buf, len, _marker: std::marker::PhantomData }
    }

    pub fn device_ptr(&self) -> *const u32 {
        self.buf.as_ptr()
    }

    pub fn device_ptr_mut(&mut self) -> *mut u32 {
        self.buf.as_mut_ptr()
    }
}

// ---- Column<BaseField> ----

impl Column<BaseField> for CudaColumn<BaseField> {
    fn zeros(len: usize) -> Self {
        let mut buf = DeviceBuffer::<u32>::alloc(len);
        buf.zero();
        Self { buf, len, _marker: std::marker::PhantomData }
    }

    unsafe fn uninitialized(len: usize) -> Self {
        let buf = DeviceBuffer::<u32>::alloc(len);
        Self { buf, len, _marker: std::marker::PhantomData }
    }

    fn to_cpu(&self) -> Vec<BaseField> {
        let host = self.buf.to_host();
        host.into_iter().map(|v| BaseField::from_u32_unchecked(v)).collect()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn at(&self, index: usize) -> BaseField {
        assert!(index < self.len);
        let mut val = [0u32; 1];
        unsafe {
            ffi::cudaMemcpy(
                val.as_mut_ptr() as *mut c_void,
                (self.buf.as_ptr() as *const u8).add(index * 4) as *const c_void,
                4, ffi::MEMCPY_D2H,
            );
        }
        BaseField::from_u32_unchecked(val[0])
    }

    fn set(&mut self, index: usize, value: BaseField) {
        assert!(index < self.len);
        let val = [value.0];
        unsafe {
            ffi::cudaMemcpy(
                (self.buf.as_mut_ptr() as *mut u8).add(index * 4) as *mut c_void,
                val.as_ptr() as *const c_void,
                4, ffi::MEMCPY_H2D,
            );
        }
    }

    fn split_at_mid(self) -> (Self, Self) {
        let half = self.len / 2;
        let cpu = self.buf.to_host();
        let left = DeviceBuffer::from_host(&cpu[..half]);
        let right = DeviceBuffer::from_host(&cpu[half..]);
        (
            Self { buf: left, len: half, _marker: std::marker::PhantomData },
            Self { buf: right, len: self.len - half, _marker: std::marker::PhantomData },
        )
    }
}

impl FromIterator<BaseField> for CudaColumn<BaseField> {
    fn from_iter<I: IntoIterator<Item = BaseField>>(iter: I) -> Self {
        let host: Vec<u32> = iter.into_iter().map(|f| f.0).collect();
        let len = host.len();
        let buf = DeviceBuffer::from_host(&host);
        Self { buf, len, _marker: std::marker::PhantomData }
    }
}

// ---- Column<SecureField> ----
// SecureField (QM31) = 4 × M31 values stored as 4 contiguous u32s per element.

impl Column<SecureField> for CudaColumn<SecureField> {
    fn zeros(len: usize) -> Self {
        let mut buf = DeviceBuffer::<u32>::alloc(len * 4);
        buf.zero();
        Self { buf, len, _marker: std::marker::PhantomData }
    }

    unsafe fn uninitialized(len: usize) -> Self {
        let buf = DeviceBuffer::<u32>::alloc(len * 4);
        Self { buf, len, _marker: std::marker::PhantomData }
    }

    fn to_cpu(&self) -> Vec<SecureField> {
        let host = self.buf.to_host();
        host.chunks_exact(4).map(|c| {
            SecureField::from_m31_array(std::array::from_fn(|i| {
                BaseField::from_u32_unchecked(c[i])
            }))
        }).collect()
    }

    fn len(&self) -> usize {
        self.len
    }

    fn is_empty(&self) -> bool {
        self.len == 0
    }

    fn at(&self, index: usize) -> SecureField {
        assert!(index < self.len);
        let mut val = [0u32; 4];
        unsafe {
            ffi::cudaMemcpy(
                val.as_mut_ptr() as *mut c_void,
                (self.buf.as_ptr() as *const u8).add(index * 16) as *const c_void,
                16, ffi::MEMCPY_D2H,
            );
        }
        SecureField::from_m31_array(std::array::from_fn(|i| {
            BaseField::from_u32_unchecked(val[i])
        }))
    }

    fn set(&mut self, index: usize, value: SecureField) {
        assert!(index < self.len);
        let arr = value.to_m31_array();
        let val: [u32; 4] = std::array::from_fn(|i| arr[i].0);
        unsafe {
            ffi::cudaMemcpy(
                (self.buf.as_mut_ptr() as *mut u8).add(index * 16) as *mut c_void,
                val.as_ptr() as *const c_void,
                16, ffi::MEMCPY_H2D,
            );
        }
    }

    fn split_at_mid(self) -> (Self, Self) {
        let half = self.len / 2;
        let cpu = self.buf.to_host();
        let left = DeviceBuffer::from_host(&cpu[..half * 4]);
        let right = DeviceBuffer::from_host(&cpu[half * 4..]);
        (
            Self { buf: left, len: half, _marker: std::marker::PhantomData },
            Self { buf: right, len: self.len - half, _marker: std::marker::PhantomData },
        )
    }
}

impl FromIterator<SecureField> for CudaColumn<SecureField> {
    fn from_iter<I: IntoIterator<Item = SecureField>>(iter: I) -> Self {
        let host: Vec<u32> = iter.into_iter().flat_map(|f| {
            let arr = f.to_m31_array();
            [arr[0].0, arr[1].0, arr[2].0, arr[3].0]
        }).collect();
        let len = host.len() / 4;
        let buf = DeviceBuffer::from_host(&host);
        Self { buf, len, _marker: std::marker::PhantomData }
    }
}

// ---- ColumnOps ----

impl ColumnOps<BaseField> for CudaBackend {
    type Column = CudaColumn<BaseField>;

    fn bit_reverse_column(column: &mut Self::Column) {
        let n = column.len;
        assert!(n.is_power_of_two());
        let log_n = n.trailing_zeros();
        unsafe {
            ffi::cuda_bit_reverse_m31(column.buf.as_mut_ptr(), log_n);
            ffi::cuda_device_sync();
        }
    }
}

impl ColumnOps<SecureField> for CudaBackend {
    type Column = CudaColumn<SecureField>;

    fn bit_reverse_column(column: &mut Self::Column) {
        // Bit-reverse each of the 4 M31 components independently
        let n = column.len;
        assert!(n.is_power_of_two());
        let log_n = n.trailing_zeros();
        // The buffer is interleaved (4 u32s per element), so we need a
        // stride-aware bit-reverse. For now, download, bit-reverse on CPU,
        // re-upload. TODO: GPU kernel for strided bit-reverse.
        let host = column.buf.to_host();
        let mut tmp = vec![0u32; n * 4];
        for i in 0..n {
            let j = bit_reverse(i, log_n as usize);
            tmp[j * 4..j * 4 + 4].copy_from_slice(&host[i * 4..i * 4 + 4]);
        }
        column.buf = DeviceBuffer::from_host(&tmp);
    }
}

// ---- Column<Blake2sHash> for Merkle tree ----

use stwo::core::vcs::blake2_hash::Blake2sHash;

impl Column<Blake2sHash> for CudaColumn<Blake2sHash> {
    fn zeros(len: usize) -> Self {
        let mut buf = DeviceBuffer::<u32>::alloc(len * 8); // 32 bytes = 8 u32s per hash
        buf.zero();
        Self { buf, len, _marker: std::marker::PhantomData }
    }

    unsafe fn uninitialized(len: usize) -> Self {
        let buf = DeviceBuffer::<u32>::alloc(len * 8);
        Self { buf, len, _marker: std::marker::PhantomData }
    }

    fn to_cpu(&self) -> Vec<Blake2sHash> {
        let host = self.buf.to_host();
        host.chunks_exact(8).map(|c| {
            let mut bytes = [0u8; 32];
            for (i, &word) in c.iter().enumerate() {
                bytes[i*4..i*4+4].copy_from_slice(&word.to_le_bytes());
            }
            Blake2sHash(bytes)
        }).collect()
    }

    fn len(&self) -> usize { self.len }

    fn at(&self, index: usize) -> Blake2sHash {
        assert!(index < self.len);
        let mut val = [0u32; 8];
        unsafe {
            ffi::cudaMemcpy(
                val.as_mut_ptr() as *mut c_void,
                (self.buf.as_ptr() as *const u8).add(index * 32) as *const c_void,
                32, ffi::MEMCPY_D2H,
            );
        }
        let mut bytes = [0u8; 32];
        for (i, &w) in val.iter().enumerate() {
            bytes[i*4..i*4+4].copy_from_slice(&w.to_le_bytes());
        }
        Blake2sHash(bytes)
    }

    fn set(&mut self, index: usize, value: Blake2sHash) {
        assert!(index < self.len);
        let mut words = [0u32; 8];
        for i in 0..8 {
            words[i] = u32::from_le_bytes([
                value.0[i*4], value.0[i*4+1], value.0[i*4+2], value.0[i*4+3]
            ]);
        }
        unsafe {
            ffi::cudaMemcpy(
                (self.buf.as_mut_ptr() as *mut u8).add(index * 32) as *mut c_void,
                words.as_ptr() as *const c_void,
                32, ffi::MEMCPY_H2D,
            );
        }
    }

    fn split_at_mid(self) -> (Self, Self) {
        let half = self.len / 2;
        let cpu = self.buf.to_host();
        let left = DeviceBuffer::from_host(&cpu[..half * 8]);
        let right = DeviceBuffer::from_host(&cpu[half * 8..]);
        (
            Self { buf: left, len: half, _marker: std::marker::PhantomData },
            Self { buf: right, len: self.len - half, _marker: std::marker::PhantomData },
        )
    }
}

impl FromIterator<Blake2sHash> for CudaColumn<Blake2sHash> {
    fn from_iter<I: IntoIterator<Item = Blake2sHash>>(iter: I) -> Self {
        let hashes: Vec<Blake2sHash> = iter.into_iter().collect();
        let len = hashes.len();
        let host: Vec<u32> = hashes.iter().flat_map(|h| {
            (0..8).map(|i| u32::from_le_bytes([
                h.0[i*4], h.0[i*4+1], h.0[i*4+2], h.0[i*4+3]
            ]))
        }).collect();
        let buf = DeviceBuffer::from_host(&host);
        Self { buf, len, _marker: std::marker::PhantomData }
    }
}

impl ColumnOps<Blake2sHash> for CudaBackend {
    type Column = CudaColumn<Blake2sHash>;

    fn bit_reverse_column(column: &mut Self::Column) {
        let n = column.len;
        assert!(n.is_power_of_two());
        let log_n = n.trailing_zeros();
        let host = column.buf.to_host();
        let mut tmp = vec![0u32; n * 8];
        for i in 0..n {
            let j = bit_reverse(i, log_n as usize);
            tmp[j * 8..j * 8 + 8].copy_from_slice(&host[i * 8..i * 8 + 8]);
        }
        column.buf = DeviceBuffer::from_host(&tmp);
    }
}

fn bit_reverse(x: usize, n_bits: usize) -> usize {
    let mut result = 0;
    let mut val = x;
    for _ in 0..n_bits {
        result = (result << 1) | (val & 1);
        val >>= 1;
    }
    result
}
