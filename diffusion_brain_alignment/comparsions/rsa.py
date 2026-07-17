import rsatoolbox

def compute_rsa(brain_data, ann_data):
    assert len(brain_data.shape) == 2, "Brain data should be a 2D array (samples x features)"
    assert len(ann_data.shape) == 2, "ANN data should be a 2D array (samples x features)"
    assert brain_data.shape[0] == ann_data.shape[0], "Number of samples must match between brain data and ANN data"
    
    bio_dataset = rsatoolbox.data.Dataset(brain_data)
    ann_dataset = rsatoolbox.data.Dataset(ann_data)
    
    bio_rdm = rsatoolbox.rdm.calc_rdm(bio_dataset, method='correlation')
    ann_rdm = rsatoolbox.rdm.calc_rdm(ann_dataset, method='correlation')
    
    rsa_result = rsatoolbox.rdm.compare(bio_rdm, ann_rdm, method='rho-a')
    
    return rsa_result
