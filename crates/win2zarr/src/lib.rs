use numpy::{PyArray1, PyArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyModule};
use win_decode::{
    decode as decode_bytes, encode as encode_seconds, ChannelSecond, SecondBlock, WinFormat,
    WinTime,
};

fn format_str(fmt: WinFormat) -> &'static str {
    match fmt {
        WinFormat::Win => "win",
        WinFormat::Win32 => "win32",
    }
}

fn parse_format(name: &str) -> PyResult<WinFormat> {
    match name.to_ascii_lowercase().as_str() {
        "win" | "win1" => Ok(WinFormat::Win),
        "win32" => Ok(WinFormat::Win32),
        other => Err(PyValueError::new_err(format!(
            "unknown WIN format {other:?}; expected 'win' or 'win32'"
        ))),
    }
}

fn time_to_dict<'py>(py: Python<'py>, t: WinTime) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("year", t.year)?;
    d.set_item("month", t.month)?;
    d.set_item("day", t.day)?;
    d.set_item("hour", t.hour)?;
    d.set_item("minute", t.minute)?;
    d.set_item("second", t.second)?;
    d.set_item("timezone", "Asia/Tokyo")?;
    Ok(d)
}

fn dict_i32(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<i32> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing {key}")))?
        .extract()
}

fn dict_u8(dict: &Bound<'_, PyDict>, key: &str) -> PyResult<u8> {
    dict.get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("missing {key}")))?
        .extract()
}

fn time_from_dict(dict: &Bound<'_, PyDict>) -> PyResult<WinTime> {
    Ok(WinTime {
        year: dict_i32(dict, "year")?,
        month: dict_u8(dict, "month")?,
        day: dict_u8(dict, "day")?,
        hour: dict_u8(dict, "hour")?,
        minute: dict_u8(dict, "minute")?,
        second: dict_u8(dict, "second")?,
    })
}

/// Decode on-disk WIN or WIN32 bytes into a nested dict of second blocks.
#[pyfunction]
fn decode_win_bytes<'py>(py: Python<'py>, data: &[u8]) -> PyResult<Bound<'py, PyDict>> {
    let decoded = decode_bytes(data).map_err(|e| PyValueError::new_err(e.to_string()))?;
    let root = PyDict::new(py);
    root.set_item("format", format_str(decoded.format))?;
    let seconds = PyList::empty(py);
    for sec in decoded.seconds {
        let sdict = PyDict::new(py);
        sdict.set_item("time", time_to_dict(py, sec.time)?)?;
        let channels = PyList::empty(py);
        for ch in sec.channels {
            let cdict = PyDict::new(py);
            cdict.set_item("channel_id", ch.channel_id)?;
            cdict.set_item("sample_rate", ch.sample_rate)?;
            let arr = PyArray1::from_vec(py, ch.samples);
            cdict.set_item("samples", arr)?;
            channels.append(cdict)?;
        }
        sdict.set_item("channels", channels)?;
        seconds.append(sdict)?;
    }
    root.set_item("seconds", seconds)?;
    Ok(root)
}

/// Encode second-block dicts to on-disk WIN or WIN32 bytes.
#[pyfunction]
fn encode_win_bytes(py: Python<'_>, format: &str, seconds: Bound<'_, PyAny>) -> PyResult<Vec<u8>> {
    let fmt = parse_format(format)?;
    let seq = seconds.downcast::<PyList>().map_err(|_| {
        PyValueError::new_err("seconds must be a list of dicts")
    })?;
    let mut blocks = Vec::with_capacity(seq.len());
    for item in seq.iter() {
        let sdict = item.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("each second must be a dict with 'time' and 'channels'")
        })?;
        let tdict = sdict
            .get_item("time")?
            .ok_or_else(|| PyValueError::new_err("second missing 'time'"))?;
        let tdict = tdict.downcast::<PyDict>().map_err(|_| {
            PyValueError::new_err("time must be a dict")
        })?;
        let time = time_from_dict(tdict)?;
        let ch_list = sdict
            .get_item("channels")?
            .ok_or_else(|| PyValueError::new_err("second missing 'channels'"))?;
        let ch_list = ch_list.downcast::<PyList>().map_err(|_| {
            PyValueError::new_err("channels must be a list")
        })?;
        let mut channels = Vec::with_capacity(ch_list.len());
        for ch in ch_list.iter() {
            let cdict = ch.downcast::<PyDict>().map_err(|_| {
                PyValueError::new_err("channel must be a dict")
            })?;
            let channel_id: u16 = cdict
                .get_item("channel_id")?
                .ok_or_else(|| PyValueError::new_err("channel missing channel_id"))?
                .extract()?;
            let sample_rate: u16 = cdict
                .get_item("sample_rate")?
                .ok_or_else(|| PyValueError::new_err("channel missing sample_rate"))?
                .extract()?;
            let samples_obj = cdict
                .get_item("samples")?
                .ok_or_else(|| PyValueError::new_err("channel missing samples"))?;
            let samples: Vec<i32> = if let Ok(arr) = samples_obj.downcast::<PyArray1<i32>>() {
                arr.to_vec()?
            } else {
                samples_obj.extract()?
            };
            channels.push(ChannelSecond {
                channel_id,
                sample_rate,
                samples,
            });
        }
        blocks.push(SecondBlock { time, channels });
    }
    let _ = py;
    encode_seconds(fmt, &blocks).map_err(|e| PyValueError::new_err(e.to_string()))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add_function(wrap_pyfunction!(decode_win_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(encode_win_bytes, m)?)?;
    Ok(())
}
