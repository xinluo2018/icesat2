import icepyx as ipx

#define function to loop through series of requests to get data
def multiple_is2_requests(earthdata_email, earthdata_uid, data_home, requests, subset, download):
    for req in requests:
        #### Download this data: (uncomment and run)
        print(req)
        region_a = ipx.Query(req['short_name'], req['spatial_extent'], req['date_range'])
        if download==True:
            region_a.earthdata_login(earthdata_uid, earthdata_email)
            region_a.download_granules(data_home, subset=subset)
        print(region_a.dataset)
        print(region_a.dates)
        print(region_a.start_time)
        print(region_a.end_time)
        print(region_a.dataset_version)
        print(region_a.spatial_extent)
        region_a.visualize_spatial_extent()