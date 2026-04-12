use serde::Serialize;


#[derive(Serialize, Debug)]
pub struct PhotoMeta {
    pub year: i16,
    pub month: i8,
    pub day: i8,
    pub minutes: i8,
    pub seconds: i8,
}
