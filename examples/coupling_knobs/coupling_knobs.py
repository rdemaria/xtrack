import numpy as np

def get_arcindex_from_name(name):
    """
    Return arc from name:
    0: arc 12
    1: arc 23
    7: arc 81
    """
    lr,aa=name.split('.')[-2][2:]
    if lr=='r':
        return (int(aa)-1)%8
    else:
        return (int(aa)-2)%8

def get_arcname_from_arcindex(ia):
    return f"a{ia+1}{(ia+2)%8}"

def get_response_from_twiss(tt,lmqs=0.32,nmqs=32):
    """
    Compute the response matrix of the RQS circuits on cmin
    lmqs: length of the MQS
    nmqs: number of the MQS in the LHC
    """
    cminus=np.zeros((2,8),dtype=float)
    smqs=0 # number of MQS slices
    for ii in range(len(tt.name)):
        name=tt.name[ii]
        if tt.name[ii].startswith('mqs.'):
            ia=get_arcindex_from_name(name)
            betx=tt.betx[ii]
            bety=tt.bety[ii]
            mux=2*np.pi*tt.mux[ii]
            muy=2*np.pi*tt.muy[ii]
            cm=np.sqrt(tt.betx[ii]*tt.bety[ii])*np.exp(1j*(mux-muy))*lmqs
            cminus[0,ia]=cm.real
            cminus[1,ia]=cm.imag
            smqs+=1
    cminus*=nmqs/smqs
    return cminus

def get_response_from_mad(mad,beam=1):
    cmr=[]
    cmi=[]
    for ia in range(8):
        lhs=f"kqs.a{ia+1}{(ia+2)%8}b{beam}"
        mad.input(f"{lhs}=1e-3;")
        t1=mad.twiss()
        cm=t1.summary.dqmin*np.exp(1j*t1.summary.dqmin_phase)
        print(lhs,mad.globals[lhs],cm)
        cmr.append(cm.real)
        cmi.append(cm.imag)
        mad.input(f"{lhs}=0;")
    return np.array([cmr,cmi]).T*1e3


def get_response_from_xtrack(tracker,beam=1):
    cmr=[]
    cmi=[]
    for ia in range(8):
        lhs=f"kqs.a{ia+1}{(ia+2)%8}b{beam}"
        tracker.vars[lhs]=1e-3
        tw=tracker.twiss(method='4d')
        cm=tw['c_minus']
        cmr.append(cm)
        cmi.append(0)
        tracker.vars[lhs]=0
    return np.array([cmr,cmi]).T*1e3



def get_knobs_from_twiss(tt,lmqs=0.32,nmqs=32):
    """
    Compute the response matrix of the RQS circuits on cminus and
    make the pseudo inverse to generate knobs weights from twiss table
    lmqs: length of the MQS
    nmqs: number of the MQS in the LHC
    """
    cminus=get_response_from_twiss(tt,lmqs=lmqs,nmqs=nmqs)
    return np.linalg.pinv(cminus)

def get_knob_defs_from_twiss(tt,beam=1,re='cmr',im='cmi'):
    """
    Generates KQS knobs in the form:
    kqs =  aa * cmr + bb * cmi
    """
    knobs=get_knobs_from_twiss(tt)
    out=[]
    for ia in range(8):
        lhs=f"kqs.a{ia+1}{(ia+2)%8}b{beam}"
        rhs=f"{knobs[ia,0]}*{re}+{knobs[ia,1]}*{im}"
        out.append((lhs,rhs))
    return out



