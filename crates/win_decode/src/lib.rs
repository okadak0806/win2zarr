//! Independent decoder for the published WIN / WIN32 waveform format.
//!
//! Implements the on-disk layout documented by ERI (`winformat(1W)`):
//! second blocks with BCD timestamps, per-channel headers, and
//! first-sample-plus-delta compression. This is **not** a copy of the
//! ERI WIN package sources (`winlib.c` / GPL-2.0).
//!
//! WIN BCD timestamps are Japan Standard Time (JST, UTC+9).

use std::fmt;

/// Detected on-disk flavour.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WinFormat {
    /// Classic WIN (2-digit BCD year, 4-byte block size prefix).
    Win,
    /// WIN32 / Hi-net style (4-digit BCD year, 16-byte second header).
    Win32,
}

/// Calendar time stored in a second-block header (JST wall clock).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct WinTime {
    pub year: i32,
    pub month: u8,
    pub day: u8,
    pub hour: u8,
    pub minute: u8,
    pub second: u8,
}

impl WinTime {
    /// Truncate to the containing minute (second = 0).
    pub fn minute_key(self) -> Self {
        Self { second: 0, ..self }
    }

    pub fn is_valid(self) -> bool {
        (1981..=2080).contains(&self.year)
            && (1..=12).contains(&self.month)
            && (1..=31).contains(&self.day)
            && self.hour <= 23
            && self.minute <= 59
            && self.second <= 59
    }
}

impl fmt::Display for WinTime {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}",
            self.year, self.month, self.day, self.hour, self.minute, self.second
        )
    }
}

/// One channel's samples for a single second.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChannelSecond {
    pub channel_id: u16,
    pub sample_rate: u16,
    pub samples: Vec<i32>,
}

/// One second of multi-channel data.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SecondBlock {
    pub time: WinTime,
    pub channels: Vec<ChannelSecond>,
}

/// Fully decoded WIN / WIN32 file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WinFile {
    pub format: WinFormat,
    pub seconds: Vec<SecondBlock>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    Truncated { offset: usize, needed: usize },
    InvalidBlockSize { offset: usize, size: u32 },
    InvalidBcd { offset: usize },
    InvalidTime { offset: usize, time: WinTime },
    InvalidSampleSize { offset: usize, sample_size: u16 },
    ChannelOverrun { offset: usize },
    Empty,
}

impl fmt::Display for DecodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DecodeError::Truncated { offset, needed } => {
                write!(
                    f,
                    "truncated WIN data at {offset}, needed {needed} more bytes"
                )
            }
            DecodeError::InvalidBlockSize { offset, size } => {
                write!(f, "invalid second-block size {size} at {offset}")
            }
            DecodeError::InvalidBcd { offset } => {
                write!(f, "invalid BCD timestamp at {offset}")
            }
            DecodeError::InvalidTime { offset, time } => {
                write!(f, "invalid timestamp {time} at {offset}")
            }
            DecodeError::InvalidSampleSize {
                offset,
                sample_size,
            } => write!(f, "unsupported sample size {sample_size} at {offset}"),
            DecodeError::ChannelOverrun { offset } => {
                write!(f, "channel block overruns second block at {offset}")
            }
            DecodeError::Empty => write!(f, "no second blocks in WIN data"),
        }
    }
}

impl std::error::Error for DecodeError {}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum EncodeError {
    Empty,
    InvalidTime(WinTime),
    InvalidSampleRate(u16),
    EmptyChannel,
    SampleRateMismatch {
        channel_id: u16,
        n_samples: usize,
        sample_rate: u16,
    },
}

impl fmt::Display for EncodeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EncodeError::Empty => write!(f, "no second blocks to encode"),
            EncodeError::InvalidTime(t) => write!(f, "invalid timestamp {t}"),
            EncodeError::InvalidSampleRate(r) => write!(f, "sample rate {r} out of 1..=4095"),
            EncodeError::EmptyChannel => write!(f, "channel has no samples"),
            EncodeError::SampleRateMismatch {
                channel_id,
                n_samples,
                sample_rate,
            } => write!(
                f,
                "channel {channel_id:04X}: {n_samples} samples but sample_rate={sample_rate}"
            ),
        }
    }
}

impl std::error::Error for EncodeError {}

fn need<'a>(data: &'a [u8], offset: usize, n: usize) -> Result<&'a [u8], DecodeError> {
    if offset + n > data.len() {
        Err(DecodeError::Truncated {
            offset,
            needed: offset + n - data.len(),
        })
    } else {
        Ok(&data[offset..offset + n])
    }
}

fn u16_be(data: &[u8], offset: usize) -> Result<u16, DecodeError> {
    let b = need(data, offset, 2)?;
    Ok(u16::from_be_bytes([b[0], b[1]]))
}

fn u32_be(data: &[u8], offset: usize) -> Result<u32, DecodeError> {
    let b = need(data, offset, 4)?;
    Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
}

fn i32_be(data: &[u8], offset: usize) -> Result<i32, DecodeError> {
    Ok(u32_be(data, offset)? as i32)
}

fn bcd_u8(byte: u8) -> Option<u8> {
    let hi = byte >> 4;
    let lo = byte & 0x0f;
    if hi > 9 || lo > 9 {
        None
    } else {
        Some(hi * 10 + lo)
    }
}

/// WIN year convention: two-digit BCD covers 1981–2080.
pub fn year_from_yy(yy: u8) -> i32 {
    if yy > 80 {
        1900 + i32::from(yy)
    } else {
        2000 + i32::from(yy)
    }
}

fn parse_win_bcd(data: &[u8], offset: usize) -> Result<WinTime, DecodeError> {
    let b = need(data, offset, 6)?;
    let yy = bcd_u8(b[0]).ok_or(DecodeError::InvalidBcd { offset })?;
    let month = bcd_u8(b[1]).ok_or(DecodeError::InvalidBcd { offset })?;
    let day = bcd_u8(b[2]).ok_or(DecodeError::InvalidBcd { offset })?;
    let hour = bcd_u8(b[3]).ok_or(DecodeError::InvalidBcd { offset })?;
    let minute = bcd_u8(b[4]).ok_or(DecodeError::InvalidBcd { offset })?;
    let second = bcd_u8(b[5]).ok_or(DecodeError::InvalidBcd { offset })?;
    let time = WinTime {
        year: year_from_yy(yy),
        month,
        day,
        hour,
        minute,
        second,
    };
    if !time.is_valid() {
        return Err(DecodeError::InvalidTime { offset, time });
    }
    Ok(time)
}

fn parse_win32_bcd(data: &[u8], offset: usize) -> Result<WinTime, DecodeError> {
    let b = need(data, offset, 8)?;
    let y1 = bcd_u8(b[0]).ok_or(DecodeError::InvalidBcd { offset })?;
    let y2 = bcd_u8(b[1]).ok_or(DecodeError::InvalidBcd { offset })?;
    let month = bcd_u8(b[2]).ok_or(DecodeError::InvalidBcd { offset })?;
    let day = bcd_u8(b[3]).ok_or(DecodeError::InvalidBcd { offset })?;
    let hour = bcd_u8(b[4]).ok_or(DecodeError::InvalidBcd { offset })?;
    let minute = bcd_u8(b[5]).ok_or(DecodeError::InvalidBcd { offset })?;
    let second = bcd_u8(b[6]).ok_or(DecodeError::InvalidBcd { offset })?;
    let time = WinTime {
        year: i32::from(y1) * 100 + i32::from(y2),
        month,
        day,
        hour,
        minute,
        second,
    };
    if !time.is_valid() {
        return Err(DecodeError::InvalidTime { offset, time });
    }
    Ok(time)
}

fn sign_extend_nibble(n: u8) -> i32 {
    let n = i32::from(n & 0x0f);
    if n & 0x08 != 0 {
        n - 16
    } else {
        n
    }
}

fn sign_extend_24(b0: u8, b1: u8, b2: u8) -> i32 {
    let v = (i32::from(b0) << 24) | (i32::from(b1) << 16) | (i32::from(b2) << 8);
    v >> 8
}

fn channel_payload_len(sample_size: u16, n_samples: u16) -> Result<usize, DecodeError> {
    if n_samples == 0 {
        return Ok(0);
    }
    let ns = n_samples as usize;
    let extra = ns - 1;
    // Selectors 0–4: remaining samples are two's-complement diffs of that width
    // (0 = nibble). Selector 5 (WIN_pkg-3) stores remaining samples as 4-byte
    // absolute i32 values, not 5-byte diffs.
    let n = match sample_size {
        0 => extra.div_ceil(2),
        1..=4 => extra * sample_size as usize,
        5 => extra * 4,
        _ => {
            return Err(DecodeError::InvalidSampleSize {
                offset: 0,
                sample_size,
            })
        }
    };
    Ok(n)
}

fn decode_diffs(
    data: &[u8],
    mut offset: usize,
    sample_size: u16,
    n_samples: u16,
    first: i32,
) -> Result<(Vec<i32>, usize), DecodeError> {
    let mut samples = Vec::with_capacity(n_samples as usize);
    samples.push(first);
    if n_samples <= 1 {
        return Ok((samples, offset));
    }
    let payload_len = channel_payload_len(sample_size, n_samples)?;
    let payload = need(data, offset, payload_len)?;
    match sample_size {
        0 => {
            let mut value = first;
            for i in 0..(n_samples as usize - 1) {
                let byte = payload[i / 2];
                let nib = if i % 2 == 0 { byte >> 4 } else { byte & 0x0f };
                value += sign_extend_nibble(nib);
                samples.push(value);
            }
        }
        1 => {
            let mut value = first;
            for &b in payload {
                value += i32::from(b as i8);
                samples.push(value);
            }
        }
        2 => {
            let mut value = first;
            for chunk in payload.chunks_exact(2) {
                value += i16::from_be_bytes([chunk[0], chunk[1]]) as i32;
                samples.push(value);
            }
        }
        3 => {
            let mut value = first;
            for chunk in payload.chunks_exact(3) {
                value += sign_extend_24(chunk[0], chunk[1], chunk[2]);
                samples.push(value);
            }
        }
        4 => {
            let mut value = first;
            for chunk in payload.chunks_exact(4) {
                value += i32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
                samples.push(value);
            }
        }
        5 => {
            // WIN_pkg-3: remaining samples are 4-byte absolute i32, not diffs.
            for chunk in payload.chunks_exact(4) {
                samples.push(i32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        other => {
            return Err(DecodeError::InvalidSampleSize {
                offset,
                sample_size: other,
            })
        }
    }
    offset += payload_len;
    Ok((samples, offset))
}

fn decode_channel_block(
    data: &[u8],
    start: usize,
    end: usize,
    win32: bool,
) -> Result<(ChannelSecond, usize), DecodeError> {
    let mut p = start;
    if win32 {
        need(data, p, 2)?;
        p += 2; // organisation / network IDs; not part of the public channel id
    }
    if p + 8 > end {
        return Err(DecodeError::ChannelOverrun { offset: start });
    }
    let channel_id = u16_be(data, p)?;
    let hdr = u16_be(data, p + 2)?;
    let sample_size = hdr >> 12;
    let sample_rate = hdr & 0x0fff;
    p += 4;
    if sample_rate == 0 {
        return Err(DecodeError::InvalidSampleSize {
            offset: start,
            sample_size,
        });
    }
    let first = i32_be(data, p)?;
    p += 4;
    let (samples, p) = decode_diffs(data, p, sample_size, sample_rate, first)?;
    if p > end {
        return Err(DecodeError::ChannelOverrun { offset: start });
    }
    Ok((
        ChannelSecond {
            channel_id,
            sample_rate,
            samples,
        },
        p,
    ))
}

fn decode_channels(
    data: &[u8],
    mut p: usize,
    end: usize,
    win32: bool,
) -> Result<Vec<ChannelSecond>, DecodeError> {
    let mut channels = Vec::new();
    while p < end {
        let (ch, next) = decode_channel_block(data, p, end, win32)?;
        if next <= p {
            return Err(DecodeError::ChannelOverrun { offset: p });
        }
        channels.push(ch);
        p = next;
    }
    Ok(channels)
}

/// Sniff WIN vs WIN32 from the start of a buffer.
pub fn detect_format(data: &[u8]) -> WinFormat {
    if data.len() >= 16 && u32::from_be_bytes([data[0], data[1], data[2], data[3]]) == 0 {
        if parse_win32_bcd(data, 4).is_ok() {
            return WinFormat::Win32;
        }
    }
    WinFormat::Win
}

fn decode_win(data: &[u8]) -> Result<Vec<SecondBlock>, DecodeError> {
    let mut offset = 0usize;
    let mut seconds = Vec::new();
    while offset < data.len() {
        let size = u32_be(data, offset)?;
        if size < 10 || size as usize > data.len() - offset {
            return Err(DecodeError::InvalidBlockSize { offset, size });
        }
        let block_end = offset + size as usize;
        let time = parse_win_bcd(data, offset + 4)?;
        let channels = decode_channels(data, offset + 10, block_end, false)?;
        seconds.push(SecondBlock { time, channels });
        offset = block_end;
    }
    Ok(seconds)
}

fn decode_win32(data: &[u8]) -> Result<Vec<SecondBlock>, DecodeError> {
    let mut offset = 4usize;
    let mut seconds = Vec::new();
    while offset < data.len() {
        if offset + 16 > data.len() {
            return Err(DecodeError::Truncated {
                offset,
                needed: offset + 16 - data.len(),
            });
        }
        let time = parse_win32_bcd(data, offset)?;
        let payload_size = u32_be(data, offset + 12)? as usize;
        let payload_start = offset + 16;
        let payload_end =
            payload_start
                .checked_add(payload_size)
                .ok_or(DecodeError::InvalidBlockSize {
                    offset,
                    size: payload_size as u32,
                })?;
        if payload_end > data.len() {
            return Err(DecodeError::Truncated {
                offset: payload_start,
                needed: payload_end - data.len(),
            });
        }
        let channels = decode_channels(data, payload_start, payload_end, true)?;
        seconds.push(SecondBlock { time, channels });
        offset = payload_end;
    }
    Ok(seconds)
}

/// Decode a complete on-disk WIN or WIN32 buffer.
pub fn decode(data: &[u8]) -> Result<WinFile, DecodeError> {
    if data.is_empty() {
        return Err(DecodeError::Empty);
    }
    let format = detect_format(data);
    let seconds = match format {
        WinFormat::Win => decode_win(data)?,
        WinFormat::Win32 => decode_win32(data)?,
    };
    if seconds.is_empty() {
        return Err(DecodeError::Empty);
    }
    Ok(WinFile { format, seconds })
}

fn to_bcd(v: u8) -> u8 {
    ((v / 10) << 4) | (v % 10)
}

fn write_win_bcd(buf: &mut Vec<u8>, time: WinTime) {
    let yy = (time.year % 100) as u8;
    buf.push(to_bcd(yy));
    buf.push(to_bcd(time.month));
    buf.push(to_bcd(time.day));
    buf.push(to_bcd(time.hour));
    buf.push(to_bcd(time.minute));
    buf.push(to_bcd(time.second));
}

fn write_win32_bcd(buf: &mut Vec<u8>, time: WinTime) {
    buf.push(to_bcd((time.year / 100) as u8));
    buf.push(to_bcd((time.year % 100) as u8));
    buf.push(to_bcd(time.month));
    buf.push(to_bcd(time.day));
    buf.push(to_bcd(time.hour));
    buf.push(to_bcd(time.minute));
    buf.push(to_bcd(time.second));
    buf.push(0);
}

fn sample_delta(prev: i32, next: i32) -> i64 {
    i64::from(next) - i64::from(prev)
}

fn choose_sample_size(samples: &[i32]) -> u16 {
    if samples.len() <= 1 {
        return 1;
    }
    let mut max_abs = 0u64;
    for w in samples.windows(2) {
        max_abs = max_abs.max(sample_delta(w[0], w[1]).unsigned_abs());
    }
    if max_abs <= 7 {
        0
    } else if max_abs <= 127 {
        1
    } else if max_abs <= 32_767 {
        2
    } else if max_abs <= 8_388_607 {
        3
    } else if max_abs <= i32::MAX as u64 {
        4
    } else {
        // Diff does not fit in a 4-byte two's-complement i32 (e.g. i32::MAX - i32::MIN).
        5
    }
}

fn encode_diffs(samples: &[i32], sample_size: u16) -> Vec<u8> {
    if samples.len() <= 1 {
        return Vec::new();
    }
    let mut out = Vec::new();
    match sample_size {
        0 => {
            let diffs: Vec<i32> = samples
                .windows(2)
                .map(|w| sample_delta(w[0], w[1]) as i32)
                .collect();
            for (i, d) in diffs.iter().enumerate() {
                let nib = (*d as u8) & 0x0f;
                if i % 2 == 0 {
                    out.push(nib << 4);
                } else {
                    let last = out.last_mut().unwrap();
                    *last |= nib;
                }
            }
        }
        1 => {
            for w in samples.windows(2) {
                out.push(sample_delta(w[0], w[1]) as i32 as i8 as u8);
            }
        }
        2 => {
            for w in samples.windows(2) {
                let d = sample_delta(w[0], w[1]) as i32 as i16;
                out.extend_from_slice(&d.to_be_bytes());
            }
        }
        3 => {
            for w in samples.windows(2) {
                let d = sample_delta(w[0], w[1]) as i32;
                out.push(((d >> 16) & 0xff) as u8);
                out.push(((d >> 8) & 0xff) as u8);
                out.push((d & 0xff) as u8);
            }
        }
        4 => {
            for w in samples.windows(2) {
                let d = sample_delta(w[0], w[1]) as i32;
                out.extend_from_slice(&d.to_be_bytes());
            }
        }
        5 => {
            // Remaining samples are 4-byte absolute i32, matching decode_diffs.
            for &sample in &samples[1..] {
                out.extend_from_slice(&sample.to_be_bytes());
            }
        }
        _ => {}
    }
    out
}

fn encode_channel(ch: &ChannelSecond, win32: bool) -> Result<Vec<u8>, EncodeError> {
    if ch.samples.is_empty() {
        return Err(EncodeError::EmptyChannel);
    }
    if !(1..=4095).contains(&ch.sample_rate) {
        return Err(EncodeError::InvalidSampleRate(ch.sample_rate));
    }
    if ch.samples.len() != ch.sample_rate as usize {
        return Err(EncodeError::SampleRateMismatch {
            channel_id: ch.channel_id,
            n_samples: ch.samples.len(),
            sample_rate: ch.sample_rate,
        });
    }
    let sample_size = choose_sample_size(&ch.samples);
    let mut buf = Vec::new();
    if win32 {
        buf.extend_from_slice(&[0, 0]);
    }
    buf.extend_from_slice(&ch.channel_id.to_be_bytes());
    let hdr = (sample_size << 12) | (ch.sample_rate & 0x0fff);
    buf.extend_from_slice(&hdr.to_be_bytes());
    buf.extend_from_slice(&ch.samples[0].to_be_bytes());
    buf.extend(encode_diffs(&ch.samples, sample_size));
    Ok(buf)
}

/// Encode second blocks as on-disk WIN or WIN32 bytes (for tests and tooling).
pub fn encode(format: WinFormat, seconds: &[SecondBlock]) -> Result<Vec<u8>, EncodeError> {
    if seconds.is_empty() {
        return Err(EncodeError::Empty);
    }
    for sec in seconds {
        if !sec.time.is_valid() {
            return Err(EncodeError::InvalidTime(sec.time));
        }
    }
    match format {
        WinFormat::Win => encode_win(seconds),
        WinFormat::Win32 => encode_win32(seconds),
    }
}

fn encode_win(seconds: &[SecondBlock]) -> Result<Vec<u8>, EncodeError> {
    let mut out = Vec::new();
    for sec in seconds {
        let mut second = Vec::new();
        write_win_bcd(&mut second, sec.time);
        for ch in &sec.channels {
            second.extend(encode_channel(ch, false)?);
        }
        let size = (4 + second.len()) as u32;
        out.extend_from_slice(&size.to_be_bytes());
        out.extend(second);
    }
    Ok(out)
}

fn encode_win32(seconds: &[SecondBlock]) -> Result<Vec<u8>, EncodeError> {
    let mut out = vec![0, 0, 0, 0];
    for sec in seconds {
        let mut payload = Vec::new();
        for ch in &sec.channels {
            payload.extend(encode_channel(ch, true)?);
        }
        let mut header = Vec::with_capacity(16);
        write_win32_bcd(&mut header, sec.time);
        header.extend_from_slice(&0u32.to_be_bytes());
        header.extend_from_slice(&(payload.len() as u32).to_be_bytes());
        debug_assert_eq!(header.len(), 16);
        out.extend(header);
        out.extend(payload);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn t(second: u8) -> WinTime {
        WinTime {
            year: 2020,
            month: 1,
            day: 2,
            hour: 3,
            minute: 4,
            second,
        }
    }

    fn sine_like(n: usize, start: i32) -> Vec<i32> {
        (0..n)
            .map(|i| start + ((i as i32 % 7) - 3) * (1 + (i as i32 % 3)))
            .collect()
    }

    #[test]
    fn roundtrip_win_4bit() {
        let samples = sine_like(100, 12);
        let file = WinFile {
            format: WinFormat::Win,
            seconds: vec![SecondBlock {
                time: t(5),
                channels: vec![ChannelSecond {
                    channel_id: 0x0A1B,
                    sample_rate: 100,
                    samples: samples.clone(),
                }],
            }],
        };
        let bytes = encode(WinFormat::Win, &file.seconds).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded.format, WinFormat::Win);
        assert_eq!(decoded.seconds[0].channels[0].samples, samples);
        assert_eq!(decoded.seconds[0].time, t(5));
        assert_eq!(decoded.seconds[0].channels[0].channel_id, 0x0A1B);
    }

    #[test]
    fn roundtrip_win_multichannel_multisecond() {
        let mut seconds = Vec::new();
        for s in 0..3u8 {
            seconds.push(SecondBlock {
                time: t(s),
                channels: vec![
                    ChannelSecond {
                        channel_id: 0x1001,
                        sample_rate: 100,
                        samples: sine_like(100, 100 + s as i32),
                    },
                    ChannelSecond {
                        channel_id: 0x1002,
                        sample_rate: 100,
                        samples: sine_like(100, -40 + s as i32 * 10),
                    },
                    ChannelSecond {
                        channel_id: 0x1003,
                        sample_rate: 100,
                        samples: sine_like(100, 7),
                    },
                ],
            });
        }
        let bytes = encode(WinFormat::Win, &seconds).unwrap();
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded.seconds.len(), 3);
        assert_eq!(decoded.seconds[2].channels.len(), 3);
        assert_eq!(
            decoded.seconds[1].channels[1].samples,
            seconds[1].channels[1].samples
        );
    }

    #[test]
    fn roundtrip_win32() {
        let samples: Vec<i32> = (0..100).map(|i| i * 3 - 50).collect();
        let seconds = vec![SecondBlock {
            time: t(10),
            channels: vec![ChannelSecond {
                channel_id: 0xABCD,
                sample_rate: 100,
                samples: samples.clone(),
            }],
        }];
        let bytes = encode(WinFormat::Win32, &seconds).unwrap();
        assert_eq!(&bytes[..4], &[0, 0, 0, 0]);
        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded.format, WinFormat::Win32);
        assert_eq!(decoded.seconds[0].channels[0].samples, samples);
        assert_eq!(decoded.seconds[0].time, t(10));
    }

    #[test]
    fn sample_size_2_and_4() {
        let big: Vec<i32> = (0..50).map(|i| i * 1000).collect();
        let huge: Vec<i32> = (0..20).map(|i| i * 1_000_000).collect();
        for (rate, samples) in [(50u16, big), (20, huge)] {
            let seconds = vec![SecondBlock {
                time: t(0),
                channels: vec![ChannelSecond {
                    channel_id: 1,
                    sample_rate: rate,
                    samples: samples.clone(),
                }],
            }];
            let bytes = encode(WinFormat::Win, &seconds).unwrap();
            let decoded = decode(&bytes).unwrap();
            assert_eq!(decoded.seconds[0].channels[0].samples, samples);
        }
    }

    #[test]
    fn nibble_sign_extend() {
        assert_eq!(sign_extend_nibble(0x07), 7);
        assert_eq!(sign_extend_nibble(0x08), -8);
        assert_eq!(sign_extend_nibble(0x0f), -1);
        assert_eq!(sign_extend_24(0xff, 0xff, 0xff), -1);
        assert_eq!(sign_extend_24(0x00, 0x00, 0x01), 1);
    }

    #[test]
    fn year_window() {
        assert_eq!(year_from_yy(81), 1981);
        assert_eq!(year_from_yy(99), 1999);
        assert_eq!(year_from_yy(0), 2000);
        assert_eq!(year_from_yy(20), 2020);
        assert_eq!(year_from_yy(80), 2080);
    }

    /// Handmade selector-5 fixture: remaining samples are 4-byte absolute i32.
    /// A following 1 Hz channel must stay aligned (extra*5 framing would desync).
    #[test]
    fn selector_5_multichannel_fixture_roundtrip() {
        let mut second = Vec::new();
        write_win_bcd(&mut second, t(5));

        second.extend_from_slice(&0x1111u16.to_be_bytes());
        let hdr_a = (5u16 << 12) | 4; // size 5, 4 samples
        second.extend_from_slice(&hdr_a.to_be_bytes());
        for sample in [10i32, -20, 1_000_000, i32::MIN] {
            second.extend_from_slice(&sample.to_be_bytes());
        }

        second.extend_from_slice(&0x2222u16.to_be_bytes());
        let hdr_b = (1u16 << 12) | 1; // size 1, 1 sample (first only)
        second.extend_from_slice(&hdr_b.to_be_bytes());
        second.extend_from_slice(&99i32.to_be_bytes());

        let mut bytes = Vec::new();
        bytes.extend_from_slice(&((4 + second.len()) as u32).to_be_bytes());
        bytes.extend(second);

        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded.seconds[0].channels.len(), 2);
        assert_eq!(
            decoded.seconds[0].channels[0].samples,
            vec![10, -20, 1_000_000, i32::MIN]
        );
        assert_eq!(decoded.seconds[0].channels[0].channel_id, 0x1111);
        assert_eq!(decoded.seconds[0].channels[1].samples, vec![99]);
        assert_eq!(decoded.seconds[0].channels[1].channel_id, 0x2222);
    }

    #[test]
    fn full_range_i32_deltas_roundtrip_without_panic() {
        let samples = vec![
            i32::MIN,
            i32::MAX,
            i32::MIN,
            0,
            i32::MAX,
            -1,
            i32::MIN,
            i32::MAX,
        ];
        assert_eq!(choose_sample_size(&samples), 5);

        let seconds = vec![SecondBlock {
            time: t(7),
            channels: vec![ChannelSecond {
                channel_id: 0x00FF,
                sample_rate: samples.len() as u16,
                samples: samples.clone(),
            }],
        }];
        let bytes = encode(WinFormat::Win, &seconds).unwrap();
        // Channel header sits after 4-byte block size + 6-byte BCD.
        let hdr = u16::from_be_bytes([bytes[12], bytes[13]]);
        assert_eq!(hdr >> 12, 5);
        // Remaining samples are absolute i32, so the second sample is stored as-is.
        let stored_second = i32::from_be_bytes(bytes[18..22].try_into().unwrap());
        assert_eq!(stored_second, i32::MAX);

        let decoded = decode(&bytes).unwrap();
        assert_eq!(decoded.seconds[0].channels[0].samples, samples);
    }
}
