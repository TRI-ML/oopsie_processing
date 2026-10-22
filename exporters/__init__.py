"""Exporters: materialize the cleaned SUBSET manifest into training formats.

materialize.py   stage the selected clean raw episodes locally in a canonical
                 camera layout (h5 + per-role mp4s), normalizing per-episode
                 camera-name variance so the two downstream exporters see one
                 consistent schema.
export_lerobot.py       -> LeRobot v2.1 dataset (pi05/openpi path).
export_webdataset.py    -> WebDataset shards via vla_foundry preprocessing.
"""
