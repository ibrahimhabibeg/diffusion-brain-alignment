import os
from pathlib import Path


from laion_fmri import load_stimuli as laion_load_stimuli
from laion_fmri.config import dataset_initialize
from laion_fmri.download import download as laion_download
from laion_fmri.download import download_stimuli
from laion_fmri.subject import load_subject

data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "laion_fmri_data"


def init_config(dataset_path=data_dir):
    os.makedirs(dataset_path, exist_ok=True)
    dataset_initialize(str(dataset_path))
    download_stimuli()


def download(subject, session):
    laion_download(
        subject=subject,
        ses=session,
        stat="effect",
        extension=["nii.gz", "tsv"],
        include_anatomical=True,
        include_stimuli=True,
        n_jobs=4,
    )


def sample_trials(subject, session, n_samples=10):
    sub = load_subject(subject=subject)

    return {
        "subject": subject,
        "session": session,
        "trials": sub.metadata[
            (sub.metadata["unique_or_shared"] == "shared") & (sub.metadata["session"] == session)
        ]
        .drop_duplicates()["session_trial"]
        .values[:n_samples]
        .tolist(),
    }


def load_stimuli(trials):
    sub = load_subject(subject=trials["subject"])
    image_names = sub.metadata["image_name"].values[trials["trials"]]
    stim = laion_load_stimuli()
    images = []
    for image_name in image_names:
        image = stim.images.get(image_name)
        images.append(image)
    return images


def load_brain_response(trials, roi=None):
    sub = load_subject(subject=trials["subject"])
    betas = sub.get_betas(session=trials["session"], roi=roi, streaming=True)
    return betas[trials["trials"]]
