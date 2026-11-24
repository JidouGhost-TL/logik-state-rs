//! Error types for Logik State GBAModPlay parsing and synthesis.

use core::fmt;

/// Errors returned when parsing Logik State GBAModPlay modules, sample banks, or audio streams.
#[derive(Debug, Clone, PartialEq, Eq)]
#[non_exhaustive]
pub enum Error {
    /// Input buffer ended before a complete header, descriptor, or payload could be read.
    UnexpectedEof,
    /// Module header or sample descriptor failed structural validation.
    InvalidHeader(&'static str),
    /// Loop range exceeded the decoded sample length.
    InvalidLoopRange {
        /// Loop start offset in samples.
        loop_start: u32,
        /// Loop length in samples.
        loop_len: u32,
        /// Total sample length in samples.
        sample_length: usize,
    },
    /// Sample index was out of range for the sample bank.
    InvalidSample(u32),
    /// Invalid synthesis or codec parameter.
    InvalidParameter(&'static str),
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnexpectedEof => {
                write!(f, "Unexpected end of data while parsing Logik State stream")
            }
            Self::InvalidHeader(msg) => write!(f, "Invalid Logik State header: {msg}"),
            Self::InvalidLoopRange {
                loop_start,
                loop_len,
                sample_length,
            } => write!(
                f,
                "Invalid Logik State loop ({loop_start}..+{loop_len}) for {sample_length}-sample waveform"
            ),
            Self::InvalidSample(id) => write!(f, "Invalid Logik State sample index {id}"),
            Self::InvalidParameter(msg) => write!(f, "Invalid Logik State parameter: {msg}"),
        }
    }
}

#[cfg(feature = "std")]
impl std::error::Error for Error {}

/// Convenience result alias for `logik-state`.
pub type Result<T> = core::result::Result<T, Error>;
