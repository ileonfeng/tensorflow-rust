#![cfg(feature = "tensorflow_unstable")]

extern crate libc;
extern crate libtensorflow_sys;

use std::any::Any;
use std::cell::RefCell;
use std::ffi::CStr;
use std::ffi::CString;
use std::ffi::NulError;
use std::fmt;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::mem;
use std::ops::Drop;
use std::os::raw;

use libtensorflow_sys as tf;

mod buffer;
pub use buffer::Buffer;

////////////////////////

thread_local!(static BUFFER_NEW_COUNT: RefCell<usize> = RefCell::new(0));

thread_local!(static BUFFER_DROP_COUNT: RefCell<usize> = RefCell::new(0));

////////////////////////

fn check_not_null<T>(p: *mut T) -> *mut T {
  assert!(!p.is_null());
  p
}

////////////////////////

macro_rules! impl_new {
  ($name: ident, $call:ident) => {
    impl $name {
      pub fn new() -> Self {
        unsafe {
          $name {
            inner: check_not_null(tf::$call()),
          }
        }
      }
    }
  }
}

////////////////////////

macro_rules! impl_drop {
  ($name: ident, $call:ident) => {
    impl Drop for $name {
      fn drop(&mut self) {
        unsafe {
          tf::$call(self.inner);
        }
      }
    }
  }
}

////////////////////////

macro_rules! c_enum {
  ($enum_name:ident { $($name:ident = $num:expr),* }) => {
    #[derive(PartialEq,Eq,PartialOrd,Ord,Debug)]
    pub enum $enum_name {
      UnrecognizedEnumValue(raw::c_uint),
      $($name),*
    }

    impl $enum_name {
      #[allow(dead_code)]
      fn from_int(value: raw::c_uint) -> $enum_name {
        match value {
          $($num => $enum_name::$name,)*
          c => $enum_name::UnrecognizedEnumValue(c),
        }
      }

      #[allow(dead_code)]
      fn to_int(&self) -> raw::c_uint {
        match self {
          &$enum_name::UnrecognizedEnumValue(c) => c,
          $(&$enum_name::$name => $num),*
        }
      }
    }

    impl ::std::fmt::Display for $enum_name {
      fn fmt(&self, f: &mut ::std::fmt::Formatter) -> ::std::fmt::Result {
        match self {
          $(&$enum_name::$name => f.write_str(stringify!($name)),)*
          &$enum_name::UnrecognizedEnumValue(c) => write!(f, "UnrecognizedEnumValue({})", c),
        }
      }
    }
  };
  ($enum_name:ident { $($name:ident = $num:expr,)* }) => {
    c_enum!($enum_name { $($name = $num),* });
  }
}

////////////////////////

c_enum!(Code {
  Ok = 0,
  Cancelled = 1,
  Unknown = 2,
  InvalidArgument = 3,
  DeadlineExceeded = 4,
  NotFound = 5,
  AlreadyExists = 6,
  PermissionDenied = 7,
  ResourceExhausted = 8,
  FailedPrecondition = 9,
  Aborted = 10,
  OutOfRange = 11,
  Unimplemented = 12,
  Internal = 13,
  Unavailable = 14,
  DataLoss = 15,
  Unauthenticated = 16,
});

////////////////////////

c_enum!(DataType {
  Float = 1,
  Double = 2,
  Int32 = 3,
  UInt8 = 4,
  Int16 = 5,
  Int8 = 6,
  String = 7,
  Complex = 8,
  Int64 = 9,
  Bool = 10,
  QInt8 = 11,
  QUInt8 = 12,
  QInt32 = 13,
  BFloat16 = 14,
  QInt16 = 15,
  QUInt16 = 16,
});

////////////////////////

pub struct Status {
  inner: *mut tf::TF_Status,
}

impl_new!(Status, TF_NewStatus);
impl_drop!(Status, TF_DeleteStatus);

impl Status {
  pub fn new_set(code: Code, msg: &str) -> std::result::Result<Status, NulError> {
    let mut status = Status::new();
    try!(status.set(code, msg));
    Ok(status)
  }

  pub fn code(&self) -> Code {
    unsafe {
      Code::from_int(tf::TF_GetCode(self.inner) as u32)
    }
  }

  pub fn is_ok(&self) -> bool {
    self.code() == Code::Ok
  }

  pub fn set(&mut self, code: Code, msg: &str) -> std::result::Result<(), NulError> {
    let message = try!(CString::new(msg)).as_ptr();
    unsafe {
      tf::TF_SetStatus(self.inner, mem::transmute(code.to_int()), message);
    }
    Ok(())
  }
}

impl Display for Status {
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    unsafe {
      try!(write!(f, "{}: ", self.code()));
      let msg = match CStr::from_ptr(tf::TF_Message(self.inner)).to_str() {
        Ok(s) => s,
        Err(_) => "<invalid UTF-8 in message>",
      };
      f.write_str(msg)
    }
  }
}

impl Debug for Status {
  fn fmt(&self, f: &mut Formatter) -> fmt::Result {
    unsafe {
      try!(write!(f, "{{inner:{:?}, ", self.inner));
      try!(write!(f, "{}: ", self.code()));
      let msg = match CStr::from_ptr(tf::TF_Message(self.inner)).to_str() {
        Ok(s) => s,
        Err(_) => "<invalid UTF-8 in message>",
      };
      try!(f.write_str(msg));
      try!(write!(f, "}}"));
      Ok(())
    }
  }
}

////////////////////////

pub struct SessionOptions {
  inner: *mut tf::TF_SessionOptions,
}

impl SessionOptions {
  pub fn set_target(&mut self, target: &str) -> std::result::Result<(), NulError> {
    let cstr = try!(CString::new(target));
    unsafe {
      tf::TF_SetTarget(self.inner, cstr.as_ptr());
    }
    Ok(())
  }

  pub fn set_config(&mut self, config: &[u8]) -> Result<()> {
    let status = Status::new();
    unsafe {
      tf::TF_SetConfig(self.inner, config.as_ptr() as *const raw::c_void, config.len(), status.inner);
    }
    if status.is_ok() {
      Ok(())
    } else {
      Err(status)
    }
  }
}

impl_new!(SessionOptions, TF_NewSessionOptions);
impl_drop!(SessionOptions, TF_DeleteSessionOptions);

////////////////////////

pub struct Session {
  inner: *mut tf::TF_Session,
}

impl Session {
  pub fn new(options: &SessionOptions) -> Result<Self> {
    let status = Status::new();
    let inner = unsafe { tf::TF_NewSession(options.inner, status.inner) };
    if inner.is_null() {
      Err(status)
    } else {
      Ok(Session {
        inner: inner,
      })
    }
  }

  pub fn close(&mut self) -> Status {
    let status = Status::new();
    unsafe {
      tf::TF_CloseSession(self.inner, status.inner);
    }
    status
  }

  pub fn extend_graph(&mut self, proto: &[u8]) -> Status {
    let status = Status::new();
    unsafe {
      tf::TF_ExtendGraph(self.inner, proto.as_ptr() as *const raw::c_void, proto.len(), status.inner);
    }
    status
  }
}

impl Drop for Session {
  fn drop(&mut self) {
    let status = Status::new();
    unsafe {
      tf::TF_DeleteSession(self.inner, status.inner);
    }
    // TODO: What do we do with the status?
  }
}

////////////////////////

pub type Result<T> = std::result::Result<T, Status>;

////////////////////////

pub trait TensorType: Default + Clone + Any {
  // TODO: Use associated constants when/if available
  fn data_type() -> DataType;
}

macro_rules! tensor_type {
  ($rust_type:ident, $tensor_type:ident) => {
    impl TensorType for $rust_type {
      fn data_type() -> DataType {
        DataType::$tensor_type
      }
    }
  }
}

tensor_type!(f32, Float);
tensor_type!(f64, Double);
tensor_type!(i32, Int32);
tensor_type!(u8, UInt8);
tensor_type!(i16, Int16);
tensor_type!(i8, Int8);
// TODO: provide type for String
// TODO: provide type for Complex
tensor_type!(i64, Int64);
tensor_type!(bool, Bool);
// TODO: provide type for QInt8
// TODO: provide type for QUInt8
// TODO: provide type for QInt32
// TODO: provide type for BFloat16
// TODO: provide type for QInt16
// TODO: provide type for QUInt16

////////////////////////

pub struct Tensor<T> {
  tensor: *mut TensorData<T>,
}

// Note that the tensor_data arg is a double pointer. We can't pass in *mut TensorData<T>, because
// in this function, we don't know what T is. We can't pass in *mut Any (or any pointer to a trait),
// because trait pointers are twice the width of native pointers. We box the *mut Any, because now
// we can cast back and forth between *mut c_void and *mut *mut Any, since *mut Any is a concrete
// type with no parameters.
unsafe extern "C" fn tensor_deallocator(_data: *mut raw::c_void,
                                 _len: libc::size_t,
                                 tensor_data: *mut raw::c_void)-> () {
  let ptr: *mut *mut Any = mem::transmute(tensor_data);
  Box::from_raw(*Box::from_raw(ptr));
}

// TODO: Replace with Iterator::product once that's stable
fn product(values: &[u64]) -> u64 {
  let mut product = 1;
  for v in values.iter() {
    product *= *v;
  }
  product
}

impl<T: TensorType + 'static> Tensor<T> {
  pub fn new(dims: &[u64]) -> Self {
    let total = product(dims);
    let data = <Buffer<T>>::new(total as usize);
    // Guaranteed safe to unwrap, because the only way for it to fail is for the
    // length of the buffer not to match the dimensions, and we created it with
    // exactly the right size.
    Self::new_with_buffer(dims, data).unwrap()
  }

  pub fn new_with_buffer(dims: &[u64], data: Buffer<T>) -> Option<Self> {
    let total = product(dims);
    if total != data.len() as u64 {
      return None
    }
    let mut dims_vec = Vec::new();
    // TODO: Use extend_from_slice once we're on Rust 1.6
    dims_vec.extend(dims.iter());
    let data_ptr = data.as_ptr() as *mut raw::c_void;
    let data_len = data.len();
    let tensor_data = Box::into_raw(Box::new(TensorData {
      inner: std::ptr::null_mut(),
      data: data,
      dims: dims_vec,
    }));
    let tensor = Tensor {
      tensor: tensor_data,
    };
    unsafe {
      // See notes on tensor_deallocator
      let destructor = Box::new(tensor_data as *mut Any);
      (*tensor.tensor).inner =
        tf::TF_NewTensor(mem::transmute(T::data_type().to_int()),
                         dims.as_ptr() as *mut i64,
                         dims.len() as i32,
                         data_ptr,
                         data_len,
                         Some(tensor_deallocator),
                         Box::into_raw(destructor) as *mut raw::c_void);
    }
    Some(tensor)
  }

  pub fn data(&self) -> &Buffer<T> {
    unsafe {
      &(*self.tensor).data
    }
  }

  pub fn data_mut(&mut self) -> &mut Buffer<T> {
    unsafe {
      &mut (*self.tensor).data
    }
  }

  pub fn dims(&self) -> &[u64] {
    unsafe {
      &(*self.tensor).dims
    }
  }

  unsafe fn detach(&mut self) {
    self.tensor = std::ptr::null_mut();
  }
}

impl<T> Drop for Tensor<T> {
  fn drop(&mut self) {
    if !self.tensor.is_null() {
      unsafe {
        tf::TF_DeleteTensor((*self.tensor).inner);
      }
    }
  }
}

struct TensorData<T> {
  inner: *mut tf::TF_Tensor,
  data: Buffer<T>,
  dims: Vec<u64>,
}

////////////////////////

#[cfg(test)]
mod tests {
  use super::*;
  use super::BUFFER_DROP_COUNT;
  use super::BUFFER_NEW_COUNT;

  fn create_session() -> Session {
    let options = SessionOptions::new();
    match Session::new(&options) {
      Ok(session) => session,
      Err(status) => panic!("Creating session failed with status: {}", status),
    }
  }

  #[test]
  fn smoke() {
    create_session();
  }

  #[test]
  fn test_close() {
    let status = create_session().close();
    assert!(status.is_ok());
  }

  #[test]
  fn test_tensor() {
    let initial_new = BUFFER_NEW_COUNT.with(|x| *x.borrow());
    let initial_drop = BUFFER_DROP_COUNT.with(|x| *x.borrow());
    {
      let mut tensor = <Tensor<f32>>::new(&[2, 3]);
      assert_eq!(tensor.data().len(), 6);
      tensor.data_mut()[0] = 1.0;
    }
    assert_eq!(BUFFER_NEW_COUNT.with(|x| *x.borrow()) - initial_new, 1);
    assert_eq!(BUFFER_DROP_COUNT.with(|x| *x.borrow()) - initial_drop, 1);
  }

  #[test]
  fn test_set_target() {
    let mut options = SessionOptions::new();
    options.set_target("local").unwrap();
  }

  #[test]
  fn test_set_config() {
    let mut options = SessionOptions::new();
    // An empty array is a valid proto, since all fields are optional.
    options.set_config(&vec![]).unwrap();
  }

  #[test]
  fn test_extend_graph() {
    let mut session = create_session();
    // An empty array is a valid proto, since all fields are optional.
    let status = session.extend_graph(&vec![]);
    assert!(status.is_ok());
  }
}
