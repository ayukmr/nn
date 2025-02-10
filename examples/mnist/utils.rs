use std::fs::File;

use anyhow::Result;
use csv::Reader;
use nalgebra::DVector;

pub fn load_data(file: File) -> Result<Vec<(usize, DVector<f32>)>> {
    let mut reader = Reader::from_reader(file);

    let data =
        reader.records()
            .filter_map(|res| {
                let record = res.unwrap();
                let row = record.iter().collect::<Vec<_>>();

                if let [str_label, str_data @ ..] = row.as_slice() {
                    let label = str_label.parse::<usize>().unwrap();

                    let data =
                        str_data.to_vec()
                            .iter()
                            .map(|v| v.parse().unwrap())
                            .collect();

                    let normalized = DVector::from_vec(data) / 255.0;

                    Some((label, normalized))
                } else {
                    None
                }
            }).collect();

    Ok(data)
}
