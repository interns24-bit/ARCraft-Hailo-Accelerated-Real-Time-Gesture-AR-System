use pyo3::prelude::*;
use opencv::{prelude::*, videoio, imgcodecs, core};

#[pyclass]
pub struct Camera {
    cap: videoio::VideoCapture,
}

#[pymethods]
impl Camera {
    #[new]
    fn new() -> Self {
        // Changed to v4l2src for your USB Camera!
        let pipeline = "v4l2src device=/dev/video0 ! videoconvert ! videoscale ! video/x-raw,width=640,height=480,format=BGR ! appsink drop=true max-buffers=1";
        let cap = videoio::VideoCapture::from_file(pipeline, videoio::CAP_GSTREAMER).unwrap();
        Camera { cap }
    }

    pub fn get_frame(&mut self) -> PyResult<Vec<u8>> {
        let mut frame = Mat::default();
        if self.cap.read(&mut frame).is_ok() && !frame.empty() {
            let mut buf = core::Vector::<u8>::new();
            imgcodecs::imencode(".jpg", &frame, &mut buf, &core::Vector::new()).unwrap();
            Ok(buf.to_vec())
        } else {
            Ok(vec![])
        }
    }
}

#[pymodule]
fn rust_eye(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Camera>()?;
    Ok(())
}
