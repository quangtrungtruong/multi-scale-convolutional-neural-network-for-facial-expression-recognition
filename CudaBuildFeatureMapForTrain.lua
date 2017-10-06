require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'cutorch'
require 'cunn'
require 'cudnn'

----------------------------------------------------------------------
-- parse command-line options
--
dname,fname = sys.fpath()
cmd = torch.CmdLine()
cmd:text()
cmd:text('Training')
cmd:text()
cmd:text('Options:')
cmd:option('-save', fname:gsub('.lua',''), 'subdirectory to save/log experiments in')
cmd:option('-network', '', 'reload pretrained network')
cmd:option('-full', false, 'use full dataset (50,000 samples)')
cmd:option('-visualize', false, 'visualize input data and weights during training')
cmd:option('-seed', 1, 'fixed input seed for repeatable experiments')

cmd:option('-optimization', 'SGD', 'optimization method: SGD | ASGD | CG | LBFGS')
cmd:option('-learningRate', 1e-2, 'learning rate at t=0')
cmd:option('-batchSize', 50, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0.95, 'momentum (SGD only)')
cmd:option('-t0', 1, 'start averaging at t0 (ASGD only), in nb of epochs')
cmd:option('-maxIter', 5, 'maximum nb of iterations for CG and LBFGS')
cmd:option('-threads', 2, 'nb of threads to use')
cmd:text()
opt = cmd:parse(arg)

-- fix seed
torch.manualSeed(opt.seed)

-- threads
torch.setnumthreads(opt.threads)
print('<torch> set nb of threads to ' .. opt.threads)

----------------------------------------------------------------------

cnn_model = torch.load('/opt/zbstudio/myprograms/CNN3/Max_NN.net')
print(cnn_model)
--print(model.modules[2]:get(1).bias)
--print(model.modules[2].output)
--print(model.modules[1].output)

----------------------------------------------------------------------
-- define model to train
-- on the 4-class classification problem
--
classes = {1, 2, 3, 4}
dir_test = '/opt/zbstudio/myprograms/Data/Fold1/Train'
seg_dir_test = '/opt/zbstudio/myprograms/Data/Fold1/Train/Label'
local channels = {'y','u' ,'v'}
-- get/create dataset
--
twidth = 32
theigh = 32
pwidth = 100 -- width of a patch centered a pixel
pheigh = 100 -- heigh of a patch centered a pixel
iwidth = 4200 --2250 -- width of input image 
iheigh = 3200 --1750 -- heigh of input image
nimage = 2 -- the number of input  
tsize_scale1 = 13072652
tsize_scale2 = 3270790
tsize_scale3 = 796488
trsize = tsize_scale1
NUM_FEATURE_MAP = 200
-- load files from folder for train
files = {}
seg_files = {}
-- Go over all files in directory. We use an iterator, paths.files().
for file in paths.files(dir_test) do
   -- We only load files that match the extension
   if file:find('jpg' .. '$') or file:find('png' .. '$') then
      -- and insert the ones we care about in our table
      table.insert(files, paths.concat(dir_test,file))
      table.insert(seg_files, paths.concat(seg_dir_test,file))
   end
end
-- Check files
if #files == 0 then
   error('given directory doesnt contain any files of type: ')
end

--normalize image by the way inserting it into the fixed image
function copyMemmoryTensor(destTensor, targetTensor)  
  targetTensor[{{},{1,destTensor:size(2)},{1, destTensor:size(3)}}] = destTensor
  return targetTensor
end

-- convert input with yuv to 10 Y channels, 3 U channels, 3 V channels
function convert3to16channels(input)
  local outTensor = torch.Tensor(16, input:size()[2], input:size()[3])
  for i=1, 3 do
    --outTensor[{{i*4-3},{},{}}] = input[i]
    outTensor[{{i*3-2},{},{}}] = input[i]
    outTensor[{{i*3-1},{},{}}] = input[i]
    outTensor[{{i*3},{},{}}] = input[i]
  end
  for i= 10, 16 do
    outTensor[{{i},{},{}}] = input[1]
  end
  return outTensor
end

-- Go over the file list:
images = {}
seg_images = {}
for i,file in ipairs(files) do
   -- load each image
   table.insert(images, image.load(file))
   print(file)
end
for i,file in ipairs(seg_files) do
   -- load each image
   table.insert(seg_images, image.load(file))
   print(file)
end

-- load dataset
inputData = {
   data_scale1 = torch.Tensor(tsize_scale1, 2),
   data_scale2 = torch.Tensor(tsize_scale2, 2),
   link = torch.IntTensor(trsize):cuda(),
   pixs_pos = torch.IntTensor(trsize, 3, 2),
   size = function() return trsize end
}

----------------------------------------------------------------------
-- preprocess/normalize train/test sets
--
-- convert input with yuv to 10 Y channels, 3 U channels, 3 V channels
function convert3to16channels(input)
  local outTensor = torch.Tensor(16, input:size()[2], input:size()[3])
  for i=1, 3 do
    --outTensor[{{i*4-3},{},{}}] = input[i]
    outTensor[{{i*3-2},{},{}}] = input[i]
    outTensor[{{i*3-1},{},{}}] = input[i]
    outTensor[{{i*3},{},{}}] = input[i]
  end
  for i= 10, 16 do
    outTensor[{{i},{},{}}] = input[1]
  end
  return outTensor
end

function getPatch(image_in, i, j)
  return image.crop(image_in, j-1-pwidth/2, i-1-pheigh/2,j-1+pwidth/2,i-1+pheigh/2)
end

-- training function
function build_feature(dataset, count_scale1, count_scale2)
   local ft_maps_scale1 = torch.Tensor(iheigh, iwidth):fill(0)
   local ft_maps_scale2 = torch.Tensor(iheigh/2, iwidth/2):fill(0)
   local vt_maps_scale1 = torch.Tensor(count_scale1, NUM_FEATURE_MAP)
   local vt_maps_scale2 = torch.Tensor(count_scale2, NUM_FEATURE_MAP)

   for t = 1, count_scale1 do
      -- get new sample
      local input_scale1 = torch.Tensor(16, pwidth, pheigh)
      local input_scale2 = torch.Tensor(16, pwidth, pheigh)
      local output_scale1 = torch.Tensor(NUM_FEATURE_MAP)
      local output_scale2 = torch.Tensor(NUM_FEATURE_MAP)
      
      input_scale1 = convert3to16channels(getPatch(scale_yuvs_train1[dataset.link[t]],dataset.data_scale1[t][1],dataset.data_scale1[t][2]))
      output1 = cnn_model.modules[2]:get(1):forward(input_scale1:cuda())
      output3 = cnn_model.modules[2]:get(3):forward(output1:cuda())
      output4 = cnn_model.modules[2]:get(4):forward(output3:cuda())
      output5 = cnn_model.modules[2]:get(5):forward(output4:cuda())
      output7 = cnn_model.modules[2]:get(7):forward(output5:cuda())
      output8 = cnn_model.modules[2]:get(8):forward(output7:cuda())
      output9 = cnn_model.modules[2]:get(9):forward(output8:cuda())
      output11 = cnn_model.modules[2]:get(11):forward(output9:cuda())
      output12 = cnn_model.modules[2]:get(12):forward(output11:cuda())
      output13 = cnn_model.modules[2]:get(13):forward(output12:cuda())
      output15 = cnn_model.modules[2]:get(15):forward(output13:cuda())
      output16 = cnn_model.modules[2]:get(16):forward(output15:cuda())
      output_scale1 = cnn_model.modules[2]:get(17):forward(output16:cuda())
      vt_maps_scale1[t] = output_scale1:double()
      ft_maps_scale1[dataset.data_scale1[t][1]][dataset.data_scale1[t][2]] = t
      
      if (t<count_scale2) then
        input_scale2 = convert3to16channels(getPatch(scale_yuvs_train2[dataset.link[t]],dataset.data_scale2[t][1],dataset.data_scale2[t][2]))
        output1 = cnn_model.modules[2]:get(1):forward(input_scale2:cuda())
        output3 = cnn_model.modules[2]:get(3):forward(output1:cuda())
        output4 = cnn_model.modules[2]:get(4):forward(output3:cuda())
        output5 = cnn_model.modules[2]:get(5):forward(output4:cuda())
        output7 = cnn_model.modules[2]:get(7):forward(output5:cuda())
        output8 = cnn_model.modules[2]:get(8):forward(output7:cuda())
        output9 = cnn_model.modules[2]:get(9):forward(output8:cuda())
        output11 = cnn_model.modules[2]:get(11):forward(output9:cuda())
        output12 = cnn_model.modules[2]:get(12):forward(output11:cuda())
        output13 = cnn_model.modules[2]:get(13):forward(output12:cuda())
        output15 = cnn_model.modules[2]:get(15):forward(output13:cuda())
        output16 = cnn_model.modules[2]:get(16):forward(output15:cuda())
        output_scale2 = cnn_model.modules[2]:get(17):forward(output16:cuda())
        vt_maps_scale2[t] = output_scale2:double()
        ft_maps_scale2[dataset.data_scale2[t][1]][dataset.data_scale2[t][2]] = t
      end
      print(t)
   end
   upsample_model2 = nn.SpatialUpSamplingNearest(2)
   local ft_maps_scale2_upsampling = upsample_model2:forward(ft_maps_scale2:reshape(1, ft_maps_scale2:size()[1], ft_maps_scale2:size()[2]))
   torch.save(paths.concat(opt.save, dataset.link[1]) ..  'maps1', ft_maps_scale1)
   torch.save(paths.concat(opt.save, dataset.link[1]) ..  'maps2', ft_maps_scale2_upsampling:reshape(ft_maps_scale2_upsampling:size()[2],ft_maps_scale2_upsampling:size()[3]))
   torch.save(paths.concat(opt.save, dataset.link[1]) ..  'list1', vt_maps_scale1)
   torch.save(paths.concat(opt.save, dataset.link[1]) ..  'list2', vt_maps_scale2)
end




print 'preprocessing data (color space + normalization)'
collectgarbage()
-- preprocess trainSet
mean_scale1 = torch.load('/opt/zbstudio/myprograms/ReviseArchitecture/CNN3/mean1')
std_scale1 = torch.load('/opt/zbstudio/myprograms/ReviseArchitecture/CNN3/std1')
mean_scale2 = torch.load('/opt/zbstudio/myprograms/ReviseArchitecture/CNN3/mean2')
std_scale2 = torch.load('/opt/zbstudio/myprograms/ReviseArchitecture/CNN3/std2')
scale_yuvs_train1 = torch.Tensor(#images, 3,iheigh,iwidth):fill(0)
scale_yuvs_train2 = torch.Tensor(#images, 3,iheigh/2,iwidth/2):fill(0)
temp_scale_yuv1 = scale_yuvs_train1:clone()
temp_scale_yuv2 = scale_yuvs_train2:clone()
for i = 1,#images do 
  -- rgb -> yuv
  local rgb = images[i]
  local yuv = image.rgb2yuv(rgb)
  --create 3 scaled tempatory images with the fixed size for all
  temp_scale_yuv1[i] = image.scale(yuv, iwidth, iheigh)
  temp_scale_yuv2[i] = image.scale(yuv, iwidth/2, iheigh/2)
  -- transform into 3 scales -- 
  scale_yuvs_train1[i] = copyMemmoryTensor(image.scale(yuv, yuv:size()[3], yuv:size()[2]), scale_yuvs_train1[i])
  scale_yuvs_train2[i] = copyMemmoryTensor(image.scale(yuv, yuv:size()[3]*0.5, yuv:size()[2]*0.5), scale_yuvs_train2[i])
end

for i,channel in ipairs(channels) do
  -- normalize each channel globally:
  scale_yuvs_train1[{ {},i,{},{} }]:add(-mean_scale1[i])
  scale_yuvs_train1[{ {},i,{},{} }]:div(std_scale1[i])
  scale_yuvs_train2[{ {},i,{},{} }]:add(-mean_scale2[i])
  scale_yuvs_train2[{ {},i,{},{} }]:div(std_scale2[i])
end

-- create dataset
for i = 5,#seg_images do
  count_scale1 = 1
  count_scale2 = 1
  local scale1_seg_image = seg_images[i]
  local scale2_seg_image = image.scale(seg_images[i], seg_images[i]:size()[3]/2, seg_images[i]:size()[2]/2)
  for h=pheigh/2+1,(scale1_seg_image:size()[2]-pheigh/2 ) do
    for w=pwidth/2+1,(scale1_seg_image:size()[3]-pwidth/2 ) do
        -- extract label per pixel for scale1
        if (math.floor(scale1_seg_image[1][h][w]*4+0.5)>=1) and (count_scale1<=trsize) then
          inputData.data_scale1[count_scale1][1]=h
          inputData.data_scale1[count_scale1][2]=w
          inputData.link[count_scale1] = i
          count_scale1 = count_scale1 + 1
        end
     
        -- extract label per pixel for scale2
       if (h<=(scale2_seg_image:size()[2]-pheigh/2)) and (w<=(scale2_seg_image:size()[3]-pwidth/2)) 
       and (math.floor(scale2_seg_image[1][h][w]*4+0.5)>=1) and (count_scale2<=trsize) then
          inputData.data_scale2[count_scale2][1]=h
          inputData.data_scale2[count_scale2][2]=w
          count_scale2 = count_scale2 + 1
        end
    end
  end
  print(inputData.link[1] .. '.jpg')
  image.save(paths.concat(opt.save, inputData.link[1]) .. '.jpg', scale_yuvs_train1[inputData.link[1]])
  build_feature(inputData, count_scale1 - 1, count_scale2 - 1)
end

for i = 1,1 do
  count_scale1 = 1
  count_scale2 = 1
  local scale1_seg_image = seg_images[i]
  local scale2_seg_image = image.scale(seg_images[i], seg_images[i]:size()[3]/2, seg_images[i]:size()[2]/2)
  for h=pheigh/2+1,(scale1_seg_image:size()[2]-pheigh/2 ) do
    for w=pwidth/2+1,(scale1_seg_image:size()[3]-pwidth/2 ) do
        -- extract label per pixel for scale1
        if (math.floor(scale1_seg_image[1][h][w]*4+0.5)>=1) and (count_scale1<=trsize) then
          inputData.data_scale1[count_scale1][1]=h
          inputData.data_scale1[count_scale1][2]=w
          inputData.link[count_scale1] = i
          count_scale1 = count_scale1 + 1
        end
     
        -- extract label per pixel for scale2
       if (h<=(scale2_seg_image:size()[2]-pheigh/2)) and (w<=(scale2_seg_image:size()[3]-pwidth/2)) 
       and (math.floor(scale2_seg_image[1][h][w]*4+0.5)>=1) and (count_scale2<=trsize) then
          inputData.data_scale2[count_scale2][1]=h
          inputData.data_scale2[count_scale2][2]=w
          count_scale2 = count_scale2 + 1
        end
    end
  end
  print(inputData.link[1] .. '.jpg')
  image.save(paths.concat(opt.save, inputData.link[1]) .. '.jpg', scale_yuvs_train1[inputData.link[1]])
  build_feature(inputData, count_scale1 - 1, count_scale2 - 1)
end