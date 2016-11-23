DEEPMASK=$PWD
img="data/dress1_200"
#limg="data/dress1_200_labeled"
limg=${img}"_labeled"
th computeProposals.lua $DEEPMASK/pretrained/deepmask -img ${img}".jpg" -limg ${limg}"_dm.jpg" # run DeepMask
th computeProposals.lua $DEEPMASK/pretrained/sharpmask -img ${img}".jpg" -limg ${limg}"_sm.jpg" # run SharpMask
#th computeProposals.lua $DEEPMASK/pretrained/sharpmask

