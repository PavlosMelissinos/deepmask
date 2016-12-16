DEEPMASK=$PWD
DDIR="/home/pmelissi/Data/sample-clothes"
img="${DDIR}/dress1_200"
#limg="${DDIR}/dress1_200_labeled"
limg="${img}_labeled"

#th computeProposals.lua $DEEPMASK/pretrained/deepmask -img "${img}.jpg" -limg "${limg}_dm.jpg" # run DeepMask
#th computeProposals.lua $DEEPMASK/pretrained/sharpmask -img ${img}".jpg" -limg ${limg}"_sm.jpg" # run SharpMask
#th computeProposals.lua $DEEPMASK/pretrained/sharpmask

#th computeProposals.lua $DEEPMASK/models -img ${img}".jpg" -limg ${limg}"_dm.jpg" # run DeepMask
#th computeProposals.lua $DEEPMASK/pretrained -img ${img}".jpg" -limg ${limg}"_dm.jpg" # run DeepMask

th computeProposals.lua models -img data/interim/train/accessories/11020382OR_13_F.png -limg 11020382OR_13_F_dm.jpg
