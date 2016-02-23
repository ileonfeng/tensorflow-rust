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
    #[derive(PartialEq,Eq,PartialOrd,Ord,Debug,Copy,Clone)]
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

fn data_type_size(t: DataType) -> usize {
  match t {
    DataType::Float => mem::size_of::<f32>(),
    DataType::Double => mem::size_of::<f64>(),
    DataType::Int32 => mem::size_of::<i32>(),
    DataType::UInt8 => mem::size_of::<u8>(),
    DataType::Int16 => mem::size_of::<i16>(),
    DataType::Int8 => mem::size_of::<i8>(),
    DataType::String => mem::size_of::<*const libc::c_char>(),
    DataType::Complex => mem::size_of::<f32>() * 2,
    DataType::Int64 => mem::size_of::<i64>(),
    DataType::Bool => mem::size_of::<u8>(),
    DataType::QInt8 => mem::size_of::<i8>(),
    DataType::QUInt8 => mem::size_of::<u8>(),
    DataType::BFloat16 => mem::size_of::<u16>(),
    DataType::QInt16 => mem::size_of::<i16>(),
    DataType::QUInt16 => mem::size_of::<u16>(),
    x => panic!("Unrecognized data type: {}", x),
  }
}

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

fn invalid_arg(msg: &str) -> Status {
  Status::new_set(Code::InvalidArgument, msg).unwrap()
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

  // Would like to take inputs as &mut [Box<AnyTensor>], but mutable slices don't have a clear() method.
  pub fn run(&mut self,
             input_names: &[&str],
             inputs: &mut Vec<Box<AnyTensor>>,
             output_tensor_names: &[&str],
             target_node_names: &[&str]) -> Result<Vec<Box<AnyTensor>>> {
    if input_names.len() != inputs.len() {
      return Err(invalid_arg(&format!("input_names.len() ({}) not equal to inputs.len() ({})", input_names.len(), inputs.len())));
    }
    // We have to keep the CStrings around in order to keep the *const c_chars valid.
    let input_name_cstrings = try!(cstring_array(input_names, "input_names"));
    let output_tensor_names_cstrings = try!(cstring_array(output_tensor_names, "output_tensor_names"));
    let target_node_names_cstrings = try!(cstring_array(target_node_names, "target_node_names"));
    let mut input_name_ptrs: Vec<*const ::libc::c_char> = input_name_cstrings.iter().map(|x| x.as_ptr()).collect();
    let mut output_tensor_name_ptrs: Vec<*const ::libc::c_char> = output_tensor_names_cstrings.iter().map(|x| x.as_ptr()).collect();
    let mut target_node_name_ptrs: Vec<*const ::libc::c_char> = target_node_names_cstrings.iter().map(|x| x.as_ptr()).collect();
    let mut tensor_ptrs: Vec<*mut tf::TF_Tensor> = inputs.iter_mut().map(|x| {
      let inner = x.inner();
      unsafe {
        x.detach();
      }
      inner
    }).collect();
    inputs.clear();
    let mut output_ptrs = Vec::with_capacity(output_tensor_names.len());
    let status = Status::new();
    unsafe {
      // TF_Run consumes the input tensors
      tf::TF_Run(self.inner,
                 input_name_ptrs.as_mut_ptr(),
                 tensor_ptrs.as_mut_ptr(),
                 tensor_ptrs.len() as i32,
                 output_tensor_name_ptrs.as_mut_ptr(),
                 output_ptrs.as_mut_ptr(),
                 output_ptrs.len() as i32,
                 target_node_name_ptrs.as_mut_ptr(),
                 target_node_name_ptrs.len() as i32,
                 status.inner);
    }
    if status.is_ok() {
      Ok(output_ptrs.iter().map(|x: &*mut tf::Struct_TF_Tensor| tensor_from_ptr(*x)).collect())
    } else {
      Err(status)
    }
  }
}

fn cstring_array(strings: &[&str], arg_name: &str) -> Result<Vec<CString>> {
  match strings.iter().map(|x| CString::new(*x)).collect() {
    Ok(x) => Ok(x),
    Err(_) => return Err(invalid_arg(&format!("{} cannot contain embedded nulls", arg_name))),
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

unsafe fn mk_tensor<T: TensorType + 'static>(inner: *mut tf::TF_Tensor, len: usize, dims: Vec<u64>) -> Box<AnyTensor> {
  let td = Box::into_raw(Box::new(TensorData {
    inner: inner,
    data: Buffer::from_ptr(tf::TF_TensorData(inner) as *mut T, len),
    dims: dims,
  }));
  Box::new(Tensor {
    tensor: td,
  })
}

fn tensor_from_ptr(inner: *mut tf::TF_Tensor) -> Box<AnyTensor> {
  unsafe {
    let data_type = DataType::from_int(mem::transmute(tf::TF_TensorType(inner)));
    let len = tf::TF_TensorByteSize(inner) / data_type_size(data_type);
    let num_dims = tf::TF_NumDims(inner);
    let mut dims = Vec::with_capacity(num_dims as usize);
    for i in 0..num_dims {
      dims.push(tf::TF_Dim(inner, i) as u64);
    }
    match data_type {
      DataType::Float => mk_tensor::<f32>(inner, len, dims),
      DataType::Double => mk_tensor::<f64>(inner, len, dims),
      DataType::Int32 => mk_tensor::<i32>(inner, len, dims),
      DataType::UInt8 => mk_tensor::<u8>(inner, len, dims),
      DataType::Int16 => mk_tensor::<i16>(inner, len, dims),
      DataType::Int8 => mk_tensor::<i8>(inner, len, dims),
      // TODO: provide type for String
      // TODO: provide type for Complex
      DataType::Int64 => mk_tensor::<i64>(inner, len, dims),
      DataType::Bool => mk_tensor::<bool>(inner, len, dims),
      // TODO: provide type for QInt8
      // TODO: provide type for QUInt8
      // TODO: provide type for QInt32
      // TODO: provide type for BFloat16
      // TODO: provide type for QInt16
      // TODO: provide type for QUInt16
      x => panic!("Unrecognized tensor type: {}", x),
    }
  }
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

pub trait AnyTensor {
  fn data_type(&self) -> DataType;
  fn inner(&self) -> *mut tf::TF_Tensor;
  unsafe fn detach(&mut self);
}

impl<T: TensorType> AnyTensor for Tensor<T> {
  fn data_type(&self) -> DataType {
    T::data_type()
  }

  fn inner(&self) -> *mut tf::TF_Tensor {
    unsafe {
      (*self.tensor).inner
    }
  }

  unsafe fn detach(&mut self) {
    Tensor::detach(self);
  }
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

  #[test]
  fn test_run() {
    let graph_proto = vec![
      0x0a, 0x2a, 0x0a, 0x01, 0x78, 0x12, 0x0b, 0x50, 0x6c, 0x61, 0x63, 0x65, 0x68, 0x6f, 0x6c, 0x64,
      0x65, 0x72, 0x2a, 0x0b, 0x0a, 0x05, 0x64, 0x74, 0x79, 0x70, 0x65, 0x12, 0x02, 0x30, 0x01, 0x2a,
      0x0b, 0x0a, 0x05, 0x73, 0x68, 0x61, 0x70, 0x65, 0x12, 0x02, 0x3a, 0x00, 0x0a, 0x30, 0x0a, 0x03,
      0x79, 0x2f, 0x79, 0x12, 0x05, 0x43, 0x6f, 0x6e, 0x73, 0x74, 0x2a, 0x0b, 0x0a, 0x05, 0x64, 0x74,
      0x79, 0x70, 0x65, 0x12, 0x02, 0x30, 0x01, 0x2a, 0x15, 0x0a, 0x05, 0x76, 0x61, 0x6c, 0x75, 0x65,
      0x12, 0x0c, 0x42, 0x0a, 0x08, 0x01, 0x12, 0x00, 0x2a, 0x04, 0x00, 0x00, 0x00, 0x40, 0x0a, 0x19,
      0x0a, 0x01, 0x79, 0x12, 0x03, 0x4d, 0x75, 0x6c, 0x1a, 0x01, 0x78, 0x1a, 0x03, 0x79, 0x2f, 0x79,
      0x2a, 0x07, 0x0a, 0x01, 0x54, 0x12, 0x02, 0x30, 0x01
    ];
    let mut session = create_session();
    let status = session.extend_graph(&graph_proto);
    assert!(status.is_ok());
    let x = Box::new(<Tensor<f32>>::new(&[2, 3])) as Box<AnyTensor>;
    let result = session.run(&vec!["x"], &mut vec![x], &vec!["y"], &vec![]).unwrap();
    assert_eq!(result.len(), 1);
  }
}
