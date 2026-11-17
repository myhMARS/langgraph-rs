//! Minimal MessagePack codec for LangGraph-compatible checkpoint documents.
//!
//! LangGraph's `JsonPlusSerializer` encodes plain JSON-shaped checkpoint tuples as
//! standard MessagePack (its custom ext types only appear for non-JSON values). This
//! codec covers the JSON subset so checkpoints can be exchanged without a dependency.

use serde_json::{Map, Value};

use crate::kernel::error::{Error, Result};

/// Encode a JSON value as MessagePack.
pub fn encode(value: &Value) -> Vec<u8> {
    let mut output = Vec::new();
    write_value(&mut output, value);
    output
}

fn write_str(output: &mut Vec<u8>, value: &str) {
    let bytes = value.as_bytes();
    let len = bytes.len();
    if len < 32 {
        output.push(0xa0 | len as u8);
    } else if len <= u8::MAX as usize {
        output.push(0xd9);
        output.push(len as u8);
    } else if len <= u16::MAX as usize {
        output.push(0xda);
        output.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        output.push(0xdb);
        output.extend_from_slice(&(len as u32).to_be_bytes());
    }
    output.extend_from_slice(bytes);
}

fn write_array_len(output: &mut Vec<u8>, len: usize) {
    if len < 16 {
        output.push(0x90 | len as u8);
    } else if len <= u16::MAX as usize {
        output.push(0xdc);
        output.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        output.push(0xdd);
        output.extend_from_slice(&(len as u32).to_be_bytes());
    }
}

fn write_map_len(output: &mut Vec<u8>, len: usize) {
    if len < 16 {
        output.push(0x80 | len as u8);
    } else if len <= u16::MAX as usize {
        output.push(0xde);
        output.extend_from_slice(&(len as u16).to_be_bytes());
    } else {
        output.push(0xdf);
        output.extend_from_slice(&(len as u32).to_be_bytes());
    }
}

fn write_value(output: &mut Vec<u8>, value: &Value) {
    if let Value::Object(object) = value
        && let Some(class) = object.get("__langgraph_type__").and_then(Value::as_str)
    {
        let mut payload = object.clone();
        payload.remove("__langgraph_type__");
        let (type_code, body_value) = if class == "Send" {
            (
                1u8,
                Value::Array(vec![
                    Value::String("langgraph.types".to_string()),
                    Value::String("Send".to_string()),
                    Value::Array(vec![
                        payload.get("node").cloned().unwrap_or(Value::Null),
                        payload.get("input").cloned().unwrap_or(Value::Null),
                    ]),
                ]),
            )
        } else {
            (
                2u8,
                Value::Array(vec![
                    Value::String("langgraph.types".to_string()),
                    Value::String(class.to_string()),
                    Value::Object(payload),
                ]),
            )
        };
        let body = {
            let mut nested = Vec::new();
            write_value(&mut nested, &body_value);
            nested
        };
        if body.len() <= u8::MAX as usize {
            output.extend_from_slice(&[0xc7, body.len() as u8, type_code]);
        } else if body.len() <= u16::MAX as usize {
            output.push(0xc8);
            output.extend_from_slice(&(body.len() as u16).to_be_bytes());
            output.push(type_code);
        } else {
            output.push(0xc9);
            output.extend_from_slice(&(body.len() as u32).to_be_bytes());
            output.push(type_code);
        }
        output.extend_from_slice(&body);
        return;
    }
    match value {
        Value::Null => output.push(0xc0),
        Value::Bool(false) => output.push(0xc2),
        Value::Bool(true) => output.push(0xc3),
        Value::Number(number) => {
            if let Some(value) = number.as_i64() {
                if (0..128).contains(&value) {
                    output.push(value as u8);
                } else if (-32..0).contains(&value) {
                    output.push((value as i8) as u8);
                } else if (0..=u8::MAX as i64).contains(&value) {
                    output.push(0xcc);
                    output.push(value as u8);
                } else if (0..=u16::MAX as i64).contains(&value) {
                    output.push(0xcd);
                    output.extend_from_slice(&(value as u16).to_be_bytes());
                } else if (0..=u32::MAX as i64).contains(&value) {
                    output.push(0xce);
                    output.extend_from_slice(&(value as u32).to_be_bytes());
                } else if (i8::MIN as i64..=i8::MAX as i64).contains(&value) {
                    output.push(0xd0);
                    output.push((value as i8) as u8);
                } else if (i16::MIN as i64..=i16::MAX as i64).contains(&value) {
                    output.push(0xd1);
                    output.extend_from_slice(&(value as i16).to_be_bytes());
                } else if (i32::MIN as i64..=i32::MAX as i64).contains(&value) {
                    output.push(0xd2);
                    output.extend_from_slice(&(value as i32).to_be_bytes());
                } else {
                    output.push(0xd3);
                    output.extend_from_slice(&value.to_be_bytes());
                }
            } else if let Some(value) = number.as_u64() {
                if value < 128 {
                    output.push(value as u8);
                } else if value <= u8::MAX as u64 {
                    output.push(0xcc);
                    output.push(value as u8);
                } else if value <= u16::MAX as u64 {
                    output.push(0xcd);
                    output.extend_from_slice(&(value as u16).to_be_bytes());
                } else if value <= u32::MAX as u64 {
                    output.push(0xce);
                    output.extend_from_slice(&(value as u32).to_be_bytes());
                } else {
                    output.push(0xcf);
                    output.extend_from_slice(&value.to_be_bytes());
                }
            } else if let Some(value) = number.as_f64() {
                output.push(0xcb);
                output.extend_from_slice(&value.to_be_bytes());
            }
        }
        Value::String(value) => write_str(output, value),
        Value::Array(values) => {
            write_array_len(output, values.len());
            for value in values {
                write_value(output, value);
            }
        }
        Value::Object(values) => {
            write_map_len(output, values.len());
            for (key, value) in values {
                write_str(output, key);
                write_value(output, value);
            }
        }
    }
}

struct Reader<'a> {
    bytes: &'a [u8],
    position: usize,
}

impl<'a> Reader<'a> {
    fn read_u8(&mut self) -> Result<u8> {
        let byte = *self
            .bytes
            .get(self.position)
            .ok_or_else(|| Error::other("msgpack: unexpected end of input"))?;
        self.position += 1;
        Ok(byte)
    }

    fn read_bytes(&mut self, len: usize) -> Result<&'a [u8]> {
        let end = self
            .position
            .checked_add(len)
            .ok_or_else(|| Error::other("msgpack: length overflow"))?;
        let slice = self
            .bytes
            .get(self.position..end)
            .ok_or_else(|| Error::other("msgpack: unexpected end of input"))?;
        self.position = end;
        Ok(slice)
    }

    fn read_u16(&mut self) -> Result<u16> {
        let bytes = self.read_bytes(2)?;
        Ok(u16::from_be_bytes([bytes[0], bytes[1]]))
    }

    fn read_u32(&mut self) -> Result<u32> {
        let bytes = self.read_bytes(4)?;
        Ok(u32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]))
    }

    fn read_u64(&mut self) -> Result<u64> {
        let bytes = self.read_bytes(8)?;
        Ok(u64::from_be_bytes([
            bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
        ]))
    }

    fn read_str(&mut self, len: usize) -> Result<Value> {
        let bytes = self.read_bytes(len)?;
        let value = std::str::from_utf8(bytes)
            .map_err(|error| Error::other(format!("msgpack: invalid utf8: {error}")))?;
        Ok(Value::String(value.to_string()))
    }

    fn read_value(&mut self) -> Result<Value> {
        let marker = self.read_u8()?;
        match marker {
            0x00..=0x7f => Ok(Value::from(marker)),
            0xe0..=0xff => Ok(Value::from(marker as i8)),
            0x80..=0x8f => self.read_map((marker & 0x0f) as usize),
            0x90..=0x9f => self.read_array((marker & 0x0f) as usize),
            0xa0..=0xbf => self.read_str((marker & 0x1f) as usize),
            0xc0 => Ok(Value::Null),
            0xc2 => Ok(Value::Bool(false)),
            0xc3 => Ok(Value::Bool(true)),
            0xca => {
                let bytes = self.read_bytes(4)?;
                let value = f32::from_be_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
                Ok(Value::from(value as f64))
            }
            0xcb => {
                let bytes = self.read_u64()?;
                Ok(Value::from(f64::from_bits(bytes)))
            }
            0xcc => Ok(Value::from(self.read_u8()?)),
            0xcd => Ok(Value::from(self.read_u16()?)),
            0xce => Ok(Value::from(self.read_u32()?)),
            0xcf => Ok(Value::from(self.read_u64()?)),
            0xd0 => Ok(Value::from(self.read_u8()? as i8)),
            0xd1 => Ok(Value::from(self.read_u16()? as i16)),
            0xd2 => Ok(Value::from(self.read_u32()? as i32)),
            0xd3 => Ok(Value::from(self.read_u64()? as i64)),
            0xd9 => {
                let len = self.read_u8()? as usize;
                self.read_str(len)
            }
            0xda => {
                let len = self.read_u16()? as usize;
                self.read_str(len)
            }
            0xdb => {
                let len = self.read_u32()? as usize;
                self.read_str(len)
            }
            0xdc => {
                let len = self.read_u16()? as usize;
                self.read_array(len)
            }
            0xdd => {
                let len = self.read_u32()? as usize;
                self.read_array(len)
            }
            0xde => {
                let len = self.read_u16()? as usize;
                self.read_map(len)
            }
            0xdf => {
                let len = self.read_u32()? as usize;
                self.read_map(len)
            }
            0xd4 => self.read_ext(1),
            0xd5 => self.read_ext(2),
            0xd6 => self.read_ext(4),
            0xd7 => self.read_ext(8),
            0xd8 => self.read_ext(16),
            0xc7 => {
                let len = self.read_u8()? as usize;
                self.read_ext(len)
            }
            0xc8 => {
                let len = self.read_u16()? as usize;
                self.read_ext(len)
            }
            0xc9 => {
                let len = self.read_u32()? as usize;
                self.read_ext(len)
            }
            other => Err(Error::other(format!(
                "msgpack: unsupported marker 0x{other:02x}"
            ))),
        }
    }

    fn read_ext(&mut self, len: usize) -> Result<Value> {
        let type_code = self.read_u8()?;
        let data = self.read_bytes(len)?;
        // LangGraph's JsonPlusSerializer wraps custom types (e.g. Interrupt) as
        // ext type 2 containing `[module, class, payload]` MessagePack.
        if type_code == 2 {
            let mut ext_reader = Reader {
                bytes: data,
                position: 0,
            };
            if let Ok(Value::Array(parts)) = ext_reader.read_value() {
                if parts.len() == 3 {
                    let class = parts[1].as_str().unwrap_or_default().to_string();
                    let payload = parts[2].clone();
                    return Ok(match payload {
                        Value::Object(mut object) => {
                            object.insert("__langgraph_type__".to_string(), Value::String(class));
                            Value::Object(object)
                        }
                        other => serde_json::json!({
                            "__langgraph_type__": class,
                            "data": other,
                        }),
                    });
                }
            }
        }
        let hex = data
            .iter()
            .map(|byte| format!("{byte:02x}"))
            .collect::<String>();
        Ok(serde_json::json!({
            "__msgpack_ext__": type_code,
            "hex": hex,
        }))
    }

    fn read_array(&mut self, len: usize) -> Result<Value> {
        let mut values = Vec::with_capacity(len.min(1024));
        for _ in 0..len {
            values.push(self.read_value()?);
        }
        Ok(Value::Array(values))
    }

    fn read_map(&mut self, len: usize) -> Result<Value> {
        let mut values = Map::with_capacity(len.min(1024));
        for _ in 0..len {
            let key = self.read_value()?;
            let Value::String(key) = key else {
                return Err(Error::other("msgpack: map keys must be strings"));
            };
            values.insert(key, self.read_value()?);
        }
        Ok(Value::Object(values))
    }
}

/// Decode MessagePack into a JSON value.
pub fn decode(bytes: &[u8]) -> Result<Value> {
    let mut reader = Reader { bytes, position: 0 };
    let value = reader.read_value()?;
    if reader.position != bytes.len() {
        return Err(Error::other("msgpack: trailing bytes after value"));
    }
    Ok(value)
}
