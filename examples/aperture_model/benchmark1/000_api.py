import xtrack as xt
from xtrack.aperture import Aperture

#context = xo.ContextCpu(omp_num_threads="auto")
lhc = xt.load("./lhc_aperture.json")
apdata = Aperture.from_line_with_madx_metadata(lhc.b1, num_profile_points=100, include_offsets=True)

from xtrack.aperture.api import ApertureAPI
ap=ApertureAPI.from_aperture_data(apdata)

ap.profiles.search("mq.*")
ap.profiles["mqwa.a5r3.b1"].get_polygon()
ap.profiles["mqwa.a5r3.b1"].plot()
ap.profiles["mqwa.a5r3.b1"].half_width=0.02
ap.profiles["mqwa.a5r3.b1"].plot()

ap.types['mqtlh.b6l3.b1']

from xtrack.aperture.api import ApertureBuilder
bld=ApertureBuilder()
bld.new_profile("aaa","RectEllipse",half_width=4.,half_height=3)
bld.new_type(...)
bld.place(...)





