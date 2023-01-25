from cpymad.madx import Madx

import coupling_knobs

mad=Madx(stdout=False)
mad.call('../../test_data/hllhc15_noerrors_nobb/sequence.madx')
mad.use('lhcb1')
t1=mad.twiss(sequence='lhcb1')
knobs=coupling_knobs.get_knob_defs_from_twiss(t1)
mad.input(f"kqs.r1b1:=kqs.a12b1;")
mad.input(f"kqs.l2b1:=kqs.a12b1;")
mad.input(f"kqs.r3b1:=kqs.a34b1;")
mad.input(f"kqs.l4b1:=kqs.a34b1;")
mad.input(f"kqs.r5b1:=kqs.a56b1;")
mad.input(f"kqs.l6b1:=kqs.a56b1;")
mad.input(f"kqs.r7b1:=kqs.a78b1;")
mad.input(f"kqs.l8b1:=kqs.a78b1;")

for lhs,rhs in knobs:
    print(f"{lhs}:={rhs};")
    mad.input(f"{lhs}:={rhs};")

import xtrack as xt
import xpart as xp

line = xt.Line.from_madx_sequence(mad.sequence.lhcb1,
                                  deferred_expressions=True)
line.particle_ref = xp.Particles(mass0=xp.PROTON_MASS_EV, q0=1,
                                 gamma0=mad.sequence.lhcb1.beam.gamma)

tracker = line.build_tracker()

tw = tracker.twiss(method='4d')

coupling_knobs.get_response_from_xtrack(tracker,beam=1)
coupling_knobs.get_response_from_mad(tracker,beam=1)








