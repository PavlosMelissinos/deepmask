require 'torch'
require 'image'
-- require 'itorch'
require('mobdebug').start()
local tds = require 'tds'
local coco0 = require 'coco'



--------------------------------------------------------------------------------
-- function: crop bbox b from inp tensor
function cropTensor(inp, b, pad)
  pad = pad or 0
  b[1], b[2] = torch.round(b[1])+1, torch.round(b[2])+1 -- 0 to 1 index
  b[3], b[4] = torch.round(b[3]), torch.round(b[4])

  print(b[1] .. ' ' .. b[2] .. ' ' .. b[3] .. ' ' .. b[4])
  local out, h, w, ind
  if #inp:size() == 3 then
    ind, out = 2, torch.Tensor(inp:size(1), b[3], b[4]):fill(pad)
  elseif #inp:size() == 2 then
    ind, out = 1, torch.Tensor(b[3], b[4]):fill(pad)
  end
  h, w = inp:size(ind), inp:size(ind+1)

  local xo1,yo1,xo2,yo2 = b[1],b[2],b[3]+b[1]-1,b[4]+b[2]-1
  local xc1,yc1,xc2,yc2 = 1,1,b[3],b[4]

  -- compute box on binary mask inp and cropped mask out
  if b[1] < 1 then xo1=1; xc1=1+(1-b[1]) end
  if b[2] < 1 then yo1=1; yc1=1+(1-b[2]) end
  if b[1]+b[3]-1 > w then xo2=w; xc2=xc2-(b[1]+b[3]-1-w) end
  if b[2]+b[4]-1 > h then yo2=h; yc2=yc2-(b[2]+b[4]-1-h) end
  local xo, yo, wo, ho = xo1, yo1, xo2-xo1+1, yo2-yo1+1
  local xc, yc, wc, hc = xc1, yc1, xc2-xc1+1, yc2-yc1+1
  if yc+hc-1 > out:size(ind)   then hc = out:size(ind  )-yc+1 end
  if xc+wc-1 > out:size(ind+1) then wc = out:size(ind+1)-xc+1 end
  if yo+ho-1 > inp:size(ind)   then ho = inp:size(ind  )-yo+1 end
  if xo+wo-1 > inp:size(ind+1) then wo = inp:size(ind+1)-xo+1 end

  out1 = out:clone()
  print('inp1: ' .. tostring(inp:size()))
  print('out1: ' .. tostring(out1:size()))
  out:narrow(ind,yc,hc); out:narrow(ind+1,xc,wc)
  inp:narrow(ind,yo,ho); inp:narrow(ind+1,xo,wo)
  image.display(inp:narrow(ind+1,yo,ho))
  print('inp2: ' .. tostring(inp:size()))


  out2 = out:clone()
  print('out2: ' .. tostring(out2:size()))

  -- image.display{inp, legend='mask out'}

  out:narrow(ind,yc,hc):narrow(ind+1,xc,wc)
  out3 = out:clone()
  print('out3: ' .. tostring(out3:size()))
  out:copy(inp:narrow(ind,yo,ho):narrow(ind+1,xo,wo))
  print('inp3: ' .. tostring(inp:size()))
  print('out: ' .. tostring(out:size()))

  -- image.display({out1, out2, out3, out})
  return out
end




function cropMask(ann, bbox, h, w, sz)
  local mask = torch.FloatTensor(sz,sz)
  local seg = ann.segmentation
  local scale = sz / bbox[3]
  local bboxS = {}
  for m = 1, #bbox do bboxS[m] = bbox[m]*scale end

  Rs = seg

  -- local polS = {}
  -- for m, segm in pairs(seg) do
  --   polS[m] = torch.DoubleTensor():resizeAs(segm):copy(segm); polS[m]:mul(scale)
  -- end
  -- local Rs = maskApi.frPoly(polS, h*scale, w*scale)
  local mo = maskApi.decode(Rs)
  local mc = cropTensor(mo, bboxS)
  mask:copy(image.scale(mc,sz,sz):gt(0.5))

  return mask
end

function jitterBox(box)
  local x, y, w, h = box[1], box[2], box[3], box[4]
  -- local xc, yc = x+w/2, y+h/2
  -- local maxDim = math.max(w,h)
  -- local scale = log2(maxDim/self.objSz)
  -- local s = scale + torch.uniform(-self.scale,self.scale)
  -- xc = xc + torch.uniform(-self.shift,self.shift)*2^s
  -- yc = yc + torch.uniform(-self.shift,self.shift)*2^s
  -- w, h = self.wSz*2^s, self.wSz*2^s
  -- return {xc-w/2, yc-h/2,w,h}
  return {x, y, w, h}
end


-- function datainds(coco, annFile )
--   assert( string.sub(annFile,-4,-1)=='json' and paths.filep(annFile) )
--   local torchFile = string.sub(annFile,1,-6) .. '.t7'
--   if not paths.filep(torchFile) then coco:__convert(annFile,torchFile) end
--   local data = torch.load(torchFile)
--   local inds = {}
--   for k,v in pairs({images='img',categories='cat',annotations='ann'}) do
--     local M = {}; inds[v..'IdsMap']=M
--     if data[k] then for i=1,data[k].id:size(1) do M[data[k].id[i]]=i end end
--   end
--   return data, inds
-- end

datadir = '/home/pmelissi/Data/yoox/interim'
-- datadir = '/home/pmelissi/Data/MS-COCO/'
split = 'train'
local annFile = string.format('%s/annotations/instances_%s.json', datadir, split)

print(annFile)

local coco = coco0.CocoApi(annFile)
-- coco.data



-- get all image ids, select one at random
local imgIds = coco:getImgIds()
local imgId = imgIds[torch.random(imgIds:numel())]

-- load image
-- img = coco:loadImgs(imgId)[1]
-- I = image.load('../images/'..dataType..'/'..img.file_name, 3)

-- load and display instance annotations
local annIds = coco:getAnnIds({imgId=imgId})
-- anns = cocoApi:loadAnns(annIds)

-- local annid = annIds[torch.random(annIds:size(1))]
maskApi = coco0.MaskApi


-- '/home/pmelissi/Data/yoox/interim/train/shirts/37836756UB_13_F.png'

function loadAnns(coco, ids )
  local anns = coco.data.annotations
  local mapper = coco.inds.annIdsMap
  -- print('annotation length')
  -- print(#anns)
  -- print('mapper length')
  -- print(#mapper)
  -- print('ids length')
  -- print(#ids)
  return coco:__load(anns, mapper, ids)
end

-- ann = coco:loadAnns(annIds)[1]
ann = loadAnns(coco, annIds)[1]
-- local bbox = ann.bbox
local bbox = jitterBox(ann.bbox)
local iSz,wSz,gSz = 160, 160 + 32, 112
-- local iSzR = iSz*(bbox[3]/wSz)
-- local xc, yc = bbox[1]+bbox[3]/2, bbox[2]+bbox[4]/2
-- local bboxInpSz = {xc-iSzR/2,yc-iSzR/2,iSzR,iSzR}

local pathImg = coco:loadImgs(ann.image_id)[1].coco_url
-- local imgName = coco:loadImgs(ann.image_id)[1].file_name
-- print(imgName)
-- local pathImg = string.format('%s/%s/%s', datadir, split, imgName)
print(tostring(ann.image_id) .. ', ' .. pathImg)
-- local inp = image.load(pathImg, 3)
-- local h, w = inp:size(2), inp:size(3)
-- inp1 = cropTensor(inp, bbox, 0.5)
-- inp1 = image.scale(inp1, wSz, wSz)

label = cropMask(ann, bbox, h, w, gSz)

local mo = maskApi.decode(ann.segmentation)

-- image.display{mo, legend='mask out'}

-- image.display{inp, legend='input image'}


-- image.display{label, legend='label'}
