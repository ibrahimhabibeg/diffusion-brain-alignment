import os
import shutil
import subprocess
from pathlib import Path


data_dir = Path(__file__).resolve().parent.parent.parent / "data" / "things" / "TVSD"
images_dir = Path(__file__).resolve().parent.parent.parent / "data" / "things" / "object_images"
images_zip = Path(__file__).resolve().parent.parent.parent / "data" / "things" / "images_THINGS.zip"

# THINGS images live in a password-protected zip on OSF. The password is public
# (it only gates acceptance of the research/non-commercial license).
IMAGES_URL = "https://osf.io/download/rdxy2/"
IMAGES_PASSWORD = b"things4all"
ZIP_PREFIX = "object_images/"

area_channels = {
    "monkeyF": {"V1": (0, 512), "IT": (512, 832), "V4": (832, 1024)},
    "monkeyN": {"V1": (0, 512), "V4": (512, 768), "IT": (768, 1024)},
}


def _datalad_env():
    """PATH that includes git-annex, even if the kernel/env doesn't (e.g. conda)."""
    env = os.environ.copy()
    if shutil.which("git-annex", path=env.get("PATH")) is None:
        extra = [p for p in ("/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin",
                             "/usr/bin") if Path(p, "git-annex").exists()]
        if extra:
            env["PATH"] = os.pathsep.join(extra + [env.get("PATH", "")])
    return env


def download(monkey="monkeyF", dataset_path=data_dir):
    """Download the TVSD neural data (via DataLad/git-annex, public, no account)."""
    dataset_path = Path(dataset_path)
    files = ["monkeyF/_logs/things_imgs.mat", f"{monkey}/THINGS_normMUA.mat"]
    if all((dataset_path / file).exists() for file in files):
        return

    env = _datalad_env()
    if shutil.which("git-annex", path=env.get("PATH")) is None:
        raise RuntimeError(
            "git-annex not found. Install it first: `brew install git-annex` (macOS) "
            "or `sudo apt install git-annex` (Linux)."
        )

    if not (dataset_path / ".datalad").exists():
        dataset_path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["datalad", "clone", "https://gin.g-node.org/paolo_papale/TVSD", str(dataset_path)],
            check=True,
            env=env,
        )
    subprocess.run(["datalad", "get", *files], cwd=dataset_path, check=True, env=env)


def download_images(zip_path=images_zip):
    """Download the THINGS stimulus images (~5 GB zip on OSF). Resumable."""
    zip_path = Path(zip_path)
    if zip_path.exists() or images_dir.exists():
        return
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["curl", "-L", "-C", "-", "--retry", "8", "--retry-delay", "30",
         "-o", str(zip_path), IMAGES_URL],
        check=True,
    )


def _decode(f, dset):
    import numpy as np

    return ["".join(chr(c) for c in np.array(f[r]).ravel()) for r in np.array(f[dset]).ravel()]


def _metadata(dataset_path=data_dir):
    import h5py

    with h5py.File(Path(dataset_path) / "monkeyF" / "_logs" / "things_imgs.mat", "r") as f:
        return {
            "train": [p.replace("\\", "/") for p in _decode(f, "train_imgs/things_path")],
            "test": [p.replace("\\", "/") for p in _decode(f, "test_imgs/things_path")],
        }


def sample_trials(monkey="monkeyF", split="train", n_samples=10, dataset_path=data_dir):
    image_names = _metadata(dataset_path)[split]
    trials = list(range(n_samples))
    return {
        "monkey": monkey,
        "split": split,
        "trials": trials,
        "image_names": [image_names[i] for i in trials],
    }


def load_stimuli(trials, images_path=images_dir, zip_path=images_zip):
    """Load the stimulus images. Reads from the extracted folder if present,
    otherwise straight from images_THINGS.zip (no extraction needed)."""
    import io
    import zipfile

    from PIL import Image
    print(zip_path)
    images_path = Path(images_path)
    use_folder = images_path.exists()
    zf = None if use_folder else (zipfile.ZipFile(zip_path) if Path(zip_path).exists() else None)

    images = []
    for image_name in trials["image_names"]:
        if use_folder:
            path = images_path / image_name
            images.append(Image.open(path).convert("RGB") if path.exists() else None)
        elif zf is not None:
            try:
                data = zf.read(ZIP_PREFIX + image_name, pwd=IMAGES_PASSWORD)
                images.append(Image.open(io.BytesIO(data)).convert("RGB"))
            except KeyError:
                images.append(None)
        else:
            images.append(None)
    if zf is not None:
        zf.close()
    return images


def load_brain_response(trials, roi=None, dataset_path=data_dir):
    import h5py
    import numpy as np

    key = "train_MUA" if trials["split"] == "train" else "test_MUA"
    path = Path(dataset_path) / trials["monkey"] / "THINGS_normMUA.mat"

    with h5py.File(path, "r") as f:
        responses = np.array(f[key])[trials["trials"]]

    if roi is None:
        return responses

    lo, hi = area_channels[trials["monkey"]][roi]
    return responses[:, lo:hi]


def load_test_repetitions(trials, roi=None, dataset_path=data_dir):
    import h5py
    import numpy as np

    path = Path(dataset_path) / trials["monkey"] / "THINGS_normMUA.mat"

    with h5py.File(path, "r") as f:
        responses = np.array(f["test_MUA_reps"])[:, trials["trials"], :]

    if roi is None:
        return responses

    lo, hi = area_channels[trials["monkey"]][roi]
    return responses[:, :, lo:hi]
