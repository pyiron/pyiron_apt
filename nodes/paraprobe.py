from pyiron_workflow import Workflow

@Workflow.wrap.as_function_node('transcoder_results')
def transcoder_job(apt_file='Si.apt', rng_file='Si.RNG', jobid=1):
    from paraprobe_parmsetup.transcoder_config import ParmsetupTranscoder, TranscodingTask
    from nodes.paraprobe_transcoder import ParaprobeTranscoder
    
    transcoder = ParmsetupTranscoder()
    transcoder_config = transcoder.load_reconstruction_and_ranging(
        recon_fpath=apt_file,
        range_fpath=rng_file,
        jobid=jobid)
    transcoder = ParaprobeTranscoder(transcoder_config)
    results = transcoder.execute()
    return results

@Workflow.wrap.as_function_node('ranger_results')
def ranger_job(transcoder_results, jobid=1):
    import os
    from paraprobe_parmsetup.ranger_config import ParmsetupRanger, ApplyExistentRanging
    ranger = ParmsetupRanger()
    ranger_config = ranger.apply_existent_ranging_definitions(
        recon_fpath=f"{transcoder_results}",
        range_fpath="", jobid=jobid)
    os.system(f"export OMP_NUM_THREADS=$MYOMP && mpiexec -n 1 paraprobe_ranger {jobid} {ranger_config}")
    ranger_results = f"PARAPROBE.Ranger.Results.SimID.{jobid}.nxs"
    return ranger_results

@Workflow.wrap.as_function_node('ranger_report')
def ranger_reporter(ranger_results):
    from paraprobe_reporter.ranger_report import ReporterRanger
    ranger_report = ReporterRanger(ranger_results)
    return ranger_report.get_summary()

@Workflow.wrap.as_function_node('surfacer_results')
def surfacer_job(transcoder_results, 
                 ranger_results, 
                 jobid=1):
    import os
    from paraprobe_parmsetup.surfacer_config import ParmsetupSurfacer, SurfaceMeshingTask
    surfacer = ParmsetupSurfacer()
    surfacer_config = surfacer.compute_convex_hull_edge_model(recon_fpath=transcoder_results,
                    range_fpath=ranger_results,
                    jobid=jobid)
    os.system(f"export OMP_NUM_THREADS=1 && mpiexec -n 1 paraprobe_surfacer {jobid} {surfacer_config}") 
    surfacer_results = f"PARAPROBE.Surfacer.Results.SimID.{jobid}.nxs"
    return surfacer_results

@Workflow.wrap.as_function_node('distancer_results')
def distancer_job(transcoder_results, 
                 ranger_results,
                 surfacer_results,
                 jobid=1):
    import os
    from paraprobe_parmsetup.distancer_config import ParmsetupDistancer, PointToTriangleSetDistancing
    distancer = ParmsetupDistancer()
    distancer_config = distancer.compute_ion_to_edge_model_distances(recon_fpath=transcoder_results,
                range_fpath=ranger_results,
                edge_fpath=surfacer_results,
                jobid=jobid)
    os.system(f"export OMP_NUM_THREADS=1 && mpiexec -n 1 paraprobe_distancer {jobid} {distancer_config}")
    distancer_results = f"PARAPROBE.Distancer.Results.SimID.{jobid}.nxs"
    return distancer_results

@Workflow.wrap.as_function_node('distancer_report')
def distancer_reporter(distancer_results, quantiles=[0.01, 0.50, 0.99], threshold=1.):
    from paraprobe_reporter.distancer_report import ReporterDistancer
    from IPython.display import Image
    distancer_report = ReporterDistancer(distancer_results, entry_id=1)
    distancer_report.get_summary(quantiles=quantiles, threshold=threshold)
    distancer_plot = distancer_report.get_ion2mesh_distance_cdf(quantile_based=True)
    return Image(filename=distancer_plot, width=500, height=500)

@Workflow.wrap.as_function_node('tessellator_results')
def tessellator_job(transcoder_results, ranger_results, jobid=1):
    import os
    from paraprobe_parmsetup.tessellator_config import ParmsetupTessellator, TessellationTask
    tessellator = ParmsetupTessellator()
    tessellator_config = tessellator.compute_complete_voronoi_tessellation(recon_fpath=transcoder_results,
                                                                          range_fpath=ranger_results,
                                                                          jobid=jobid)
    os.system(f"export OMP_NUM_THREADS=1 && mpiexec -n 1 paraprobe_tessellator {jobid} {tessellator_config}")
    tessellator_results = f"PARAPROBE.Tessellator.Results.SimID.{jobid}.nxs"
    return tessellator_results

@Workflow.wrap.as_function_node('tessellator_report')
def tessellator_reporter(tessellator_results, jobid=1, entry_id=1, quantile_based=True,
                        height=500, width=500):
    from paraprobe_reporter.tessellator_report import ReporterTessellator
    from IPython.display import Image
    tessellator_report = ReporterTessellator(tessellator_results, entry_id=entry_id)
    tessellator_plot = tessellator_report.get_cell_volume_cdf(task_id=1, 
                                                                     quantile_based=quantile_based)
    tessellator_image = Image(filename=tessellator_plot, width=width, height=height)
    return tessellator_image

@Workflow.wrap.as_function_node('voxel_center')
def get_center(transcoder_results):
    import numpy as np
    import h5py
    with h5py.File(transcoder_results, "r") as h5r:
        xyz = h5r["/entry1/atom_probe/reconstruction/reconstructed_positions"][:, :]
        aabb = np.zeros([3, 2], np.float32)
        center = [0., 0., 0.]  # np.zeros([3], np.float32)
        for i in np.arange(0, 3):
            aabb[i, 0] = np.min(xyz[:, i])
            aabb[i, 1] = np.max(xyz[:, i])
            center[i] = 0.5 * (aabb[i, 0] + aabb[i, 1])
    return center

@Workflow.wrap.as_function_node('selector_results')
def selector_job(transcoder_results, ranger_results, center, boxdims=[10., 10., 40.], jobid=1):
    from paraprobe_utils.primscontinuum import RoiRotatedCuboid, RoiRotatedCylinder, RoiSphere
    from paraprobe_parmsetup.selector_config import ParmsetupSelector, RoiSelectionTask
    import os
    import numpy as np
    selector = ParmsetupSelector()
    task = RoiSelectionTask()
    task.load_reconstruction(recon_fpath=transcoder_results)
    task.load_ranging(iontypes_fpath=ranger_results)
    task.flt.add_spatial_filter(primitive_list=[RoiRotatedCuboid(center=center, boxdims=boxdims)])
    task.flt.add_evaporation_id_filter(lival=(0, 2, np.iinfo(np.uint32).max))  # each second ion only
    selector.add_task(task)
    selector_config = selector.configure(jobid)
    os.system(f"export OMP_NUM_THREADS=1 && mpiexec -n 1 paraprobe_selector {jobid} {selector_config}")
    selector_results = f"PARAPROBE.Selector.Results.SimID.{jobid}.nxs"
    return selector_results

@Workflow.wrap.as_function_node('selector_report')
def selector_reporter(selector_results):
    from paraprobe_reporter.selector_report import ReporterSelector
    selector_report = ReporterSelector(selector_results)
    return selector_report.get_summary()    
    
@Workflow.wrap.as_function_node('spatstat_results')
def spatstat_job(transcoder_results, ranger_results, distancer_results, jobid=1, d_edge=0.160, d_feature=0.678):
    from paraprobe_parmsetup.spatstat_config import ParmsetupSpatstat, SpatstatTask
    import os
    spatstat = ParmsetupSpatstat()
    # define two tasks, first instantiate a task object
    task = SpatstatTask()
    task.load_reconstruction(
        recon_fpath=transcoder_results)
    task.load_ranging(
        iontypes_fpath=ranger_results)
    # add filter as is exemplified for paraprobe-selector
    task.load_ion_to_edge_distances(
        fpath=distancer_results,
        dset_name=f"/entry1/point_to_triangle/distance",
        d_edge=d_edge)
    task.load_ion_to_feature_distances(
        fpath=distancer_results,
        dset_name=f"/entry1/point_to_triangle/distance",
        d_feature=d_feature)
    task.ion_types_source(method="resolve_all")
    task.ion_types_target(method="resolve_all")
    # either or
    task.set_knn(kth=1, binwidth=0.01, rmax=2.)
    task.set_rdf(binwidth=0.01, rmax=2.)
    spatstat.add_task(task)
    spatstat_config = spatstat.configure(jobid)
    os.system(f"export OMP_NUM_THREADS=1 && mpiexec -n 1 paraprobe_spatstat {jobid} {spatstat_config}")
    spatstat_results = f"PARAPROBE.Spatstat.Results.SimID.{jobid}.nxs"
    return spatstat_results

@Workflow.wrap.as_function_node('spatstat_plot')
def spatstat_plot(spatstat_results, jobid=1, entry_id=1, task_id=1, rho=1):
    from paraprobe_reporter.spatstat_report import ReporterSpatstat
    from IPython.display import Image
    spatstat_report = ReporterSpatstat(spatstat_results, entry_id=entry_id)
    spatstat_report.get_knn(task_id=task_id)
    spatstat_report.get_rdf(1, normalizer=1./rho) # dont forget to make a proper estimate for rho!
    return Image(filename="PARAPROBE.Spatstat.Results.SimID.1.nxs.EntryId.1.TaskId.1.Knn.Pdf.png", width=500, height=500)

@Workflow.wrap.as_function_node('nanochem_results')
def nanochem_job(transcoder_results, ranger_results, surfacer_results, distancer_results, element='Cr', 
                 voxel_edge_length=1., sigma=1., pixel_size=2, 
                 iso_start=0.01, iso_stop=0.05, iso_num=5, jobid=1):
    import numpy as np
    import os
    from paraprobe_parmsetup.nanochem_config import ParmsetupNanochem, Delocalization
    
    nanochem = ParmsetupNanochem()
    task = Delocalization()
    task.load_reconstruction(recon_fpath=transcoder_results)
    task.load_ranging(iontypes_fpath=ranger_results)
    task.load_edge_model(fpath=surfacer_results,
        vertices_dset_name="/entry1/point_set_wrapping/alpha_complex1/triangle_set/triangles/vertices",
        facet_indices_dset_name="/entry1/point_set_wrapping/alpha_complex1/triangle_set/triangles/faces")
    task.load_ion_to_edge_distances(
        fpath=distancer_results,
        dset_name=f"/entry1/point_to_triangle/distance")
    
    task.set_delocalization_input(method="compute")
    task.set_delocalization_normalization(method="composition")  # normalize to atomic fraction (at.-%)
    task.set_delocalization_whitelist(method="resolve_element", nuclide_hash=[element], charge_state=[])  # iso-surface defined by all atoms of (molecular) ions with Cr in it ...
    task.set_delocalization_gridresolutions(length=[voxel_edge_length])  # nm, list of voxel edge length, for each length one analysis
    task.set_delocalization_kernel(sigma=[sigma], size=pixel_size)  # nm and pixel respectively
    task.set_delocalization_isosurfaces(phi=np.linspace(start=iso_start, stop=iso_stop, num=iso_num, endpoint=True)) # isosurface starting at 4 at.-% in steps of 1 at.-% until 5 at.-%
    # task.set_delocalization_isosurfaces(phi=[0.04]) # isosurface only for 4 at.-% 
    task.set_delocalization_edge_handling(method="keep_edge_triangles")
    task.set_delocalization_edge_threshold(1.)
    task.report_fields_and_gradients(True)
    task.report_triangle_soup(True)
    task.report_objects(True)
    task.report_objects_properties(True)
    task.report_objects_geometry(True)
    task.report_objects_optimal_bounding_box(True)
    task.report_objects_ions(True)
    task.report_objects_edge_contact(True)
    # combinatorial closure of objects that are not watertight
    task.report_proxies(False)
    task.report_proxies_properties(False)
    task.report_proxies_geometry(False)
    task.report_proxies_optimal_bounding_box(False)
    task.report_proxies_ions(False)
    task.report_proxies_edge_contact(False)
    
    nanochem.add_task(task)
    nanochem_config = nanochem.configure(jobid)  # , verbose=True)
    os.system(f"export OMP_NUM_THREADS=1 && mpiexec -n 1 paraprobe_nanochem {jobid} {nanochem_config}")
    nanochem_results = f"PARAPROBE.Nanochem.Results.SimID.{jobid}.nxs"
    return nanochem_results

@Workflow.wrap.as_function_node('nanochem_VolOverIsoComposition', 'nanochem_NumberOverIsoComposition')
def nanochem_reporter(nanochem_results, width=500, height=500):
    from paraprobe_reporter.nanochem_report import ReporterNanochem
    from IPython.display import Image
    nanochem_report = ReporterNanochem(nanochem_results)
    nanochem_report.get_delocalization(deloc_task_id=1)
    nanochem_report.get_isosurface_objects_volume_and_number_over_isovalue(deloc_task_id=1)
    n_plot_voc = "PARAPROBE.Nanochem.Results.SimID.1.nxs.EntryId.1.DelocTaskId.1.VolOverIsoComposition.png"
    n_plot_noc = "PARAPROBE.Nanochem.Results.SimID.1.nxs.EntryId.1.DelocTaskId.1.NumberOverIsoComposition.png"
    return Image(filename=n_plot_voc, width=width, height=height), Image(filename=n_plot_noc, width=width, height=height)