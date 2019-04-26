set frame_importance_file [ lindex $argv 0 ]
set pdb [ lindex $argv 1 ]
set xtc_file [ lindex $argv 2 ]
#set outfile [ lindex $argv 1 ]
mol new $pdb type pdb waitfor all
mol addfile $xtc_file type {xtc} first 0 last -1 step 1 waitfor all

proc color_scheme {} {
  set color_start [colorinfo num]
  display update off
  for {set i 0} {$i < 1024} {incr i} {
    # from WHITE to BLUE
    set r [expr 1-$i/1024.] ;  set g [expr 1-$i/1024.] ; set b 1
    color change rgb [expr $i + $color_start ] $r $g $b }
  display update on }

color_scheme

display update on
axes location off
display projection orthographic
display resetview
color Display Background white
display shadows on
display ambientocclusion on
mol delrep 0 top
mol selection {protein}
mol representation NewCartoon
mol color Beta
mol addrep top
mol modmaterial 0 top AOChalky
mol scaleminmax top 0 0 1
rotate x by -90
#rotate y by -45
rotate z by 180
# [atomselect top "all"] moveby {0 4 0}
scale by 1.65
material change ambient AOChalky 0.1
material change outline AOChalky 1.4
display depthcue off
display rendermode GLSL
display update


#################Animation
####For Movie rendering##########33
#Smooth with 5 frames per window to remove some thermal vibrations
mol smoothrep 0 0 5
menu vmdmovie on
#############LOAD data into user field
##Below from https://www.ks.uiuc.edu/Research/vmd/mailing_list/vmd-l/5001.html
#The ke file is just a long list of ke 
#for each atom, and each frame is cancatinated to the file. This 
#script then applies each ke to each atom, frame after frame
#script then applies each ke to each atom, frame after frame
set frame_importances [open $frame_importance_file r] 
set numframes [molinfo top get numframes] 
set numatoms [molinfo top get numatoms] 
mol color User
mol modcolor 0 top User 
mol colupdate 0 top 1 
mol scaleminmax top 0 0.0 1.0 
for {set i 1} {$i<($numframes)} {incr i} {   
  animate goto $i 
  [atomselect top "all"] set user 0
  if {$i%100==0} {
    puts "Setting User data for frame $i/$numframes ..." 
  }
  for {set j 0} {$j<($numatoms)} {incr j} { 
    set fi [gets $frame_importances] 
    if {[string first "#" $fi] != -1} {
      #puts "Found next frame, leaving loop"
      if {$j==0} {
        #We are reading the header of this frame, just read the next line
        set fi [gets $frame_importances] 
       } else {
        puts "#Number of atoms in toplogy does not match number of atoms in the file and we reached the next frame -> will go to next frame"
        break 
       }
    }
    if {$fi<0.5} {continue}
    set atomsel [atomselect top "index $j" frame $i] 
    $atomsel set user $fi
    $atomsel delete 
  } 
} 
#render Tachyon [format $pdb.dat] 
#/usr/local/lib/vmd/tachyon_LINUXAMD64 $pdb.dat -format TARGA -res 630 1000 -o $outfile.tga
#exit
