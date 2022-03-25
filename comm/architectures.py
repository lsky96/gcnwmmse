import comm.gcnwmmse
import comm.reference_wmmse_unrolls


def get_model(model_name, *args, **kwargs):

    if model_name == "GCNWMMSE":
        return comm.gcnwmmse.GCNWMMSE(*args, **kwargs)
    elif model_name == "GCNWMMSE_Adhoc":
        return comm.gcnwmmse.GCNWMMSE_SISOAdhoc(*args, **kwargs)

    elif model_name == "UWMMSE":
        return comm.reference_wmmse_unrolls.UWMMSE(*args, **kwargs)
    elif model_name == "IADNN":
        return comm.reference_wmmse_unrolls.IAIDNN(*args, **kwargs)
    elif model_name == "UnfoldedPGD":
        return comm.reference_wmmse_unrolls.UnfoldedPGD(*args, **kwargs)

    else:
        raise ValueError("Unknown model entered.")
