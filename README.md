# icesat2
Scripts for icesat2 data processing

## structure
_utils and utils_main are functional api and comman line api, respectively._
- data access:   
  data_acess.ipynb  
- data reading and write out:   
  readatl03.py; readatl06.py    
- data organization:   
  split_file.py  
  merge_files.py; subset_file.py  
- preprocessing:   
  tile_match.py; make_grid.py   
- processing:   
  interp.py; xover.py 



### to do
- move the scripts from helper.py to specific code.
- 

### Acknowledgement
Many thanks to following codes that help us a lot in building this codebase:    
- [2020_ICESat-2_Hackweek_Tutorials](https://github.com/ICESAT-2HackWeek/2020_ICESat-2_Hackweek_Tutorials)  
- [captoolkit](https://github.com/fspaolo/captoolkit)



