import os

from laion_fmri.config import dataset_initialize
from laion_fmri.download import download as laion_download
from laion_fmri.download import download_stimuli

script_dir = os.path.dirname(os.path.abspath(__file__))


def init_config(dataset_path=os.path.join(script_dir, "../../../data/laion_fmri_data")):
    os.makedirs(dataset_path, exist_ok=True)
    dataset_initialize(dataset_path)
    download_stimuli()


def download(subject, session):
    laion_download(
        subject=subject,
        ses=session,
        stat="effect",
        extension="nii.gz",
        include_anatomical=True,
        include_stimuli=True,
        n_jobs=4,
    )
