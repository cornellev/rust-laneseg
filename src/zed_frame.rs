//! ZED wire format, vendored from `rust-autonomy-sensors/crates/zed-reader`.
//!
//! `zed-reader`'s build script needs the ZED SDK, so depending on the crate
//! directly would make this node unbuildable anywhere but the car. This is a
//! byte-for-byte copy of its header layout and key expressions; keep it in
//! sync (or move both onto a shared SDK-free codec crate once one exists).
use anyhow::{Result, ensure};

#[cfg(not(target_endian = "little"))]
compile_error!("ZED frames require a little-endian target");

pub const WIDTH: usize = 672;
pub const HEIGHT: usize = 376;
pub const BGRA_LEN: usize = WIDTH * HEIGHT * 4;
pub const DEPTH_LEN: usize = WIDTH * HEIGHT;
pub const FRAME_HEADER_LEN: usize = 32;
pub const IMAGE_FRAME_LEN: usize = FRAME_HEADER_LEN + BGRA_LEN;
pub const DEPTH_FRAME_LEN: usize = FRAME_HEADER_LEN + DEPTH_LEN * size_of::<f32>();
pub const IMAGE_KEY_EXPR: &str = "zed/zed2/image_left_vga_bgra";
pub const DEPTH_KEY_EXPR: &str = "zed/zed2/depth_vga_f32";
pub const FRAME_ENCODING: &str = "application/vnd.cornellev.zed-frame";

const MAGIC: &[u8; 4] = b"ZED1";
const VERSION: u8 = 1;
pub const IMAGE_KIND: u8 = 1;
pub const DEPTH_KIND: u8 = 2;

pub struct ZedFrame<'a> {
    pub frame_id: u64,
    pub timestamp_ns: u64,
    pub data: &'a [u8],
}

pub struct ZedDepthFrame<'a> {
    pub frame_id: u64,
    pub timestamp_ns: u64,
    data: &'a [u8],
}

impl ZedDepthFrame<'_> {
    pub fn values(&self) -> impl ExactSizeIterator<Item = f32> + '_ {
        self.data
            .as_chunks::<4>()
            .0
            .iter()
            .copied()
            .map(f32::from_le_bytes)
    }
}

type Header = (u64, u64, u32, u32);

pub fn write_header(out: &mut [u8], kind: u8, header: Header) {
    out[..4].copy_from_slice(MAGIC);
    out[4..8].copy_from_slice(&[VERSION, kind, 0, 0]);
    out[8..16].copy_from_slice(&header.0.to_le_bytes());
    out[16..24].copy_from_slice(&header.1.to_le_bytes());
    out[24..28].copy_from_slice(&header.2.to_le_bytes());
    out[28..32].copy_from_slice(&header.3.to_le_bytes());
}

fn field<const N: usize>(payload: &[u8], offset: usize) -> [u8; N] {
    payload[offset..offset + N]
        .try_into()
        .expect("validated frame length")
}

fn decode_frame(payload: &[u8], kind: u8, len: usize) -> Result<(Header, &[u8])> {
    ensure!(payload.len() == len, "frame has wrong length");
    ensure!(&payload[..4] == MAGIC, "invalid frame magic");
    ensure!(payload[4] == VERSION, "unsupported frame version");
    ensure!(
        payload[5] == kind && payload[6..8] == [0, 0],
        "invalid frame header"
    );
    let header = (
        u64::from_le_bytes(field(payload, 8)),
        u64::from_le_bytes(field(payload, 16)),
        u32::from_le_bytes(field(payload, 24)),
        u32::from_le_bytes(field(payload, 28)),
    );
    ensure!(
        (header.2 as usize, header.3 as usize) == (WIDTH, HEIGHT),
        "unexpected frame dimensions"
    );
    Ok((header, &payload[FRAME_HEADER_LEN..]))
}

pub fn decode_image_frame(payload: &[u8]) -> Result<ZedFrame<'_>> {
    let ((frame_id, timestamp_ns, _, _), data) =
        decode_frame(payload, IMAGE_KIND, IMAGE_FRAME_LEN)?;
    Ok(ZedFrame {
        frame_id,
        timestamp_ns,
        data,
    })
}

pub fn decode_depth_frame(payload: &[u8]) -> Result<ZedDepthFrame<'_>> {
    let ((frame_id, timestamp_ns, _, _), data) =
        decode_frame(payload, DEPTH_KIND, DEPTH_FRAME_LEN)?;
    Ok(ZedDepthFrame {
        frame_id,
        timestamp_ns,
        data,
    })
}

/// Encode a frame the way `zed-reader` does. Used by the replay tool and tests.
pub fn encode_frame(kind: u8, frame_id: u64, timestamp_ns: u64, body: &[u8]) -> Vec<u8> {
    let mut out = vec![0; FRAME_HEADER_LEN + body.len()];
    write_header(
        &mut out,
        kind,
        (frame_id, timestamp_ns, WIDTH as u32, HEIGHT as u32),
    );
    out[FRAME_HEADER_LEN..].copy_from_slice(body);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip() {
        let image = encode_frame(IMAGE_KIND, 42, 123, &vec![17; BGRA_LEN]);
        let decoded = decode_image_frame(&image).unwrap();
        assert_eq!((decoded.frame_id, decoded.timestamp_ns, decoded.data[0]), (42, 123, 17));

        let depth: Vec<u8> = (0..DEPTH_LEN).flat_map(|_| 1.25_f32.to_le_bytes()).collect();
        let depth = encode_frame(DEPTH_KIND, 42, 123, &depth);
        assert_eq!(decode_depth_frame(&depth).unwrap().values().next(), Some(1.25));
        assert!(decode_image_frame(&depth).is_err());
    }
}
