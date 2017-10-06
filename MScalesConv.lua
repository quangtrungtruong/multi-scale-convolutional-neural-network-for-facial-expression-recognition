----------------------------------------------------------------------
-- This script shows how to train different models 
-- dataset, using multiple optimization techniques (SGD, ASGD, CG)
--
-- This script demonstrates a classical example of training
-- well-known models (convnet, MLP, logistic regression)
-- on a -class classification problem.
--
-- It illustrates several points:
-- 1/ description of the model
-- 2/ choice of a loss function (criterion) to minimize
-- 3/ creation of a dataset as a simple Lua table
-- 4/ description of training and test procedures
----------------------------------------------------------------------

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
-- define model to train
-- on the 4-class classification problem
--
classes = {1, 2, 3, 4}
dir = '/opt/zbstudio/myprograms/Data/Fold1/Train'
seg_dir = '/opt/zbstudio/myprograms/Data/Fold1/Train/Label'
dir_test = '/opt/zbstudio/myprograms/Data/Fold1/Test'
seg_dir_test = '/opt/zbstudio/myprograms/Data/Fold1/Test/Label'
ext = 'jpg'

----------------------------------------------------------------------
-- get/create dataset
pwidth = 100 -- width of a patch centered a pixel
pheigh = 100 -- heigh of a patch centered a pixel
scale = 1
iwidth = 4200 --2250 -- width of input image 
iheigh = 3200 --1750 -- heigh of input image
nimage = 2 -- the number of input
limit_class1 = 2050913
limit_class2 = 820429
limit_class3 = 569208
limit_class4 = 959249
local channels = {'y','u' ,'v'}


if opt.full then
   trsize = 42356--nimage*iwidth*iheight
   tesize = 100000--nimage*iwidth*iheight/2
else
   trsize = limit_class1 + limit_class2 + limit_class3 + limit_class4
   tesize = 200000
end
print('trsize: ', trsize, ' tesize: ', tesize)

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

-- load files from folder for train
files = {}
seg_files = {}
-- Go over all files in directory. We use an iterator, paths.files().
for file in paths.files(dir) do
   -- We only load files that match the extension
   if file:find('jpg' .. '$') or file:find('png' .. '$') then
      -- and insert the ones we care about in our table
      table.insert(files, paths.concat(dir,file))
      table.insert(seg_files, paths.concat(seg_dir,file))
   end
end
-- Check files
if #files == 0 then
   error('given directory doesnt contain any files of type: ' .. ext)
end

-- Go over the file list:
images = {}
seg_images = {}
for i,file in ipairs(files) do
   -- load each image
   --table.insert(images, image.scale(image.load(file), iwidth, iheigh))
   local img = image.load(file)
   local img1 = image.scale(img, img:size()[3]*scale, img:size()[2]*scale)
   table.insert(images, img1)
   print(file)
end
for i,file in ipairs(seg_files) do
   -- load each image
   --table.insert(seg_images, image.scale(image.load(file), iwidth, iheigh))
   local img = image.load(file)
   local img1 = image.scale(img, img:size()[3]*scale, img:size()[2]*scale)
   table.insert(seg_images, img1)
   print(file)
end

-- load dataset
trainData = {
   data_scale1 = torch.Tensor(trsize, 2),
   data_scale2 = torch.Tensor(trsize, 2),
   link = torch.IntTensor(trsize),
   labels = torch.IntTensor(trsize),
   size = function() return trsize end
}
----------------------------------------------------------------------
-- preprocess/normalize train/test sets
print '<trainer> preprocessing data (color space + normalization)'
collectgarbage()
-- preprocess trainSet
-- Normalize each channel, and store mean/std
-- per channel. These values are important, as they are part of
-- the trainable parameters. At test time, test data will be normalized
-- using these values.
mean_scale1 = {}
std_scale1 = {}
mean_scale2 = {}
std_scale2 = {}
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
  mean_scale1[i] = temp_scale_yuv1[{ {},i,{},{} }]:mean()
  std_scale1[i] = temp_scale_yuv1[{ {},i,{},{} }]:std()
  mean_scale2[i] = temp_scale_yuv2[{ {},i,{},{} }]:mean()
  std_scale2[i] = temp_scale_yuv2[{ {},i,{},{} }]:std()
   
  scale_yuvs_train1[{ {},i,{},{} }]:add(-mean_scale1[i])
  scale_yuvs_train1[{ {},i,{},{} }]:div(std_scale1[i])
  scale_yuvs_train2[{ {},i,{},{} }]:add(-mean_scale2[i])
  scale_yuvs_train2[{ {},i,{},{} }]:div(std_scale2[i])
  --save mean, std data
  print('Mean, std for scale 1 of channel', i, ':', mean_scale1[i], std_scale1[i])
  print('Mean, std for scale 2 of channel', i, ':', mean_scale2[i], std_scale2[i])
end

torch.save(paths.concat(opt.save, 'mean1'), mean_scale1)
torch.save(paths.concat(opt.save, 'mean2'), mean_scale2)
torch.save(paths.concat(opt.save, 'std1'), std_scale1)
torch.save(paths.concat(opt.save, 'std2'), std_scale2)

-- save yuv channels of image after pr-processing for visualization
for i = 1,#images do 
  local dir = paths.concat(opt.save, i)
  image.save(dir .. '1.jpg', scale_yuvs_train1[i])
  image.save(dir .. '2.jpg', scale_yuvs_train2[i])
end

count_class1 = 0
count_class2 = 0
count_class3 = 0
count_class4 = 0
-- create dataset
local count = 1
for i = 1,#seg_images do 
  modifyImg = scale_yuvs_train2[i]:clone()
  local chosenZone = 1
  local scale_seg_image = image.scale(seg_images[i], seg_images[i]:size()[3]/2, seg_images[i]:size()[2]/2)
  for h=pheigh/2+1,(scale_seg_image:size()[2]-pheigh/2 ) do
    for w=pwidth/2+1,(scale_seg_image:size()[3]-pwidth/2 ) do
     local lab = math.floor(scale_seg_image[1][h][w]*4+0.5)
     if (lab==1 and count_class1<limit_class1) or (lab==2 and count_class2<limit_class2) or (lab==3 and count_class3<limit_class3) or (lab==4 and count_class4<limit_class4) then   
       if (math.floor(scale_seg_image[1][h][w]*4+0.5)>=1) and (count<=trsize) then
          trainData.data_scale1[count][1]=h*2
          trainData.data_scale1[count][2]=w*2
          trainData.data_scale2[count][1]=h
          trainData.data_scale2[count][2]=w
          trainData.labels[count]=math.floor(scale_seg_image[1][h][w]*4+0.5)
          if trainData.labels[count] == 1 then
            count_class1 = count_class1 + 1
          elseif trainData.labels[count] == 2 then
            count_class2 = count_class2 + 1
          elseif trainData.labels[count] == 3 then
            count_class3 = count_class3 + 1
          else
            count_class4 = count_class4 + 1
          end
          modifyImg[1][h][w] = trainData.labels[count]/4
          modifyImg[2][h][w] = trainData.labels[count]/4
          modifyImg[3][h][w] = trainData.labels[count]/4
          
          trainData.link[count] = i
          count = count + 1
        end
      end
    end
  end 
  image.save(paths.concat(opt.save, i) .. 'modifiedTrainImage.jpg', modifyImg)
  image.save(paths.concat(opt.save, i) .. 'TrainImage.jpg', scale_yuvs_train2[i])
end

print ('count train ', count)
print ('Class1: ', count_class1, ' class2: ', count_class2, ' class3: ', count_class3, ' class4: ', count_class4)

if opt.network == '' then
   -- define model to train
   model = nn.ParallelTable()
   conv_bank1 = nn.Sequential()

    ------------------------------------------------------------
    -- convolutional network
    ------------------------------------------------------------
    -- stage 1 : mean+std normalization -> filter bank -> squashing -> max pooling
    -- define connect table
    connect_table = torch.Tensor(16,2)
    for i=1,16 do
      if i<11 then
        connect_table[i][1] = 1
        connect_table[i][2] = i
      elseif i<14 then
        connect_table[i][1] = 2
        connect_table[i][2] = i
      else
        connect_table[i][1] = 3
        connect_table[i][2] = i
      end
    end
    -- CNN architecure
    --conv_bank1:add(nn.SpatialConvolutionMap(connect_table, 7, 7)) 
    conv_bank1:add(nn.SpatialConvolution(16,64, 9, 9))
    conv_bank1:add(nn.SpatialBatchNormalization(64, 1e-4))
    conv_bank1:add(nn.ReLU(true))
    --conv_bank1:add(nn.Dropout(0.3))
    conv_bank1:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    -- stage 2 : filter bank -> squashing -> max pooling
    --conv_bank1:add(nn.SpatialConvolutionMap(nn.tables.random(16, 64, 8), 7, 7))
    conv_bank1:add(nn.SpatialConvolution(64, 512, 7, 7)) 
    conv_bank1:add(nn.SpatialBatchNormalization(512, 1e-4))
    conv_bank1:add(nn.ReLU(true))
    --conv_bank1:add(nn.Dropout(0.4))
    conv_bank1:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    -- stage 3 : standard 2-layer neural network
    --conv_bank1:add(nn.SpatialConvolutionMap(nn.tables.random(64, 256, 32), 7, 7))
    conv_bank1:add(nn.SpatialConvolution(512, 1028, 7, 7)) 
    --conv_bank1:add(nn.Dropout(0.4))
    conv_bank1:add(nn.SpatialBatchNormalization(1028, 1e-4))
    conv_bank1:add(nn.ReLU(true))
    conv_bank1:add(nn.SpatialMaxPooling(2, 2, 2, 2))
    -- stage 4 : standard 2-layer neural network
    conv_bank1:add(nn.SpatialConvolution(1028, 200, 7, 7)) 
    --conv_bank1:add(nn.Dropout(0.4))
    conv_bank1:add(nn.SpatialBatchNormalization(200, 1e-4))
    conv_bank1:add(nn.ReLU(true))
    -- state 5: fully connected
    conv_bank1:add(nn.Reshape(200))
    conv_bank1:add(nn.ReLU(true))
    conv_bank1:add(nn.Linear(200,#classes))
    conv_bank1:add(nn.LogSoftMax())
    model:add(conv_bank1)
      ------------------------------------------------------------
      -- model for scale 1, 2
      -- make a copy that shares the weights and biases
    conv_bank2=conv_bank1:clone('weight','bias','gradWeight','gradBias');
    model:add(conv_bank2)
    --model = torch.load('/opt/zbstudio/myprograms/CVPR17/CNN/CNN2/NN (copy).net')
    model = model:cuda()
    cudnn.convert(model, cudnn)
    conv_bank1 = model.modules[1]
    conv_bank2 = model.modules[2]
end

-- retrieve parameters and gradients
parameters,gradParameters = model:getParameters()

-- verbose
print('using model:')
print(model)

----------------------------------------------------------------------
-- loss function: negative log-likelihood
local total = count_class1 + count_class2 + count_class3 + count_class4

weight = torch.Tensor{total/count_class1, total/count_class2, total/count_class3, total/count_class4}
--weight = torch.Tensor{1/count_class1, 1/count_class2, 1/count_class3, 1/count_class4}
print('weight ', weight)
criterion1 = nn.ClassNLLCriterion(weight):cuda()
criterion2 = nn.ClassNLLCriterion(weight):cuda()
-- load files from folder for train
files_test = {}
seg_files_test = {}
-- Go over all files in directory. We use an iterator, paths.files().
for file in paths.files(dir_test) do
   -- We only load files that match the extension
   if file:find('jpg' .. '$') or file:find('png' .. '$') then
      -- and insert the ones we care about in our table
      table.insert(files_test, paths.concat(dir_test,file))
      table.insert(seg_files_test, paths.concat(seg_dir_test,file))
   end
end
-- Check files
if #files_test == 0 then
   error('given directory doesnt contain any files of type: ' .. ext)
end

-- Go over the file list:
images_test = {}
seg_images_test = {}
for i,file in ipairs(files_test) do
   -- load each image
   --table.insert(images_test, image.scale(image.load(file), iwidth, iheigh))
   local img = image.load(file)
   local img1 = image.scale(img, img:size()[3]*scale, img:size()[2]*scale)
   table.insert(images_test, img1)
end
for i,file in ipairs(seg_files_test) do
   -- load each image
   --table.insert(seg_images_test, image.scale(image.load(file), iwidth, iheigh))
   local img = image.load(file)
   local img1 = image.scale(img, img:size()[3]*scale, img:size()[2]*scale)
   table.insert(seg_images_test, img1)
end

-- load dataset
testData = {
   data_scale1 = torch.Tensor(tesize, 2),
   data_scale2 = torch.Tensor(tesize, 2),
   link = torch.Tensor(tesize),
   labels = torch.IntTensor(tesize),
   size = function() return tesize end
}
----------------------------------------------------------------------
-- preprocess/normalize train/test sets
--

print '<tester> preprocessing data (color space + normalization)'
collectgarbage()

-- preprocess testSet
--normalize the test image to its fixed size 3000*4000
scale_yuvs_test1 = torch.Tensor(#images_test, 3,iheigh,iwidth)
scale_yuvs_test2 = torch.Tensor(#images_test, 3,iheigh/2,iwidth/2)
for i = 1,#images_test do 
   -- rgb -> yuv
  local rgb = images_test[i]
  local yuv = image.rgb2yuv(rgb)
  scale_yuvs_test1[i] = copyMemmoryTensor(image.scale(yuv, yuv:size()[3], yuv:size()[2]), scale_yuvs_test1[i])
  scale_yuvs_test2[i] = copyMemmoryTensor(image.scale(yuv, yuv:size()[3]*0.5, yuv:size()[2]*0.5), scale_yuvs_test2[i])
end
d = 0
for i,channel in ipairs(channels) do
   -- normalize each channel globally:
   scale_yuvs_test1[{ {},i,{},{} }]:add(-mean_scale1[i])
   scale_yuvs_test1[{ {},i,{},{} }]:div(std_scale1[i])
   scale_yuvs_test2[{ {},i,{},{} }]:add(-mean_scale2[i])
   scale_yuvs_test2[{ {},i,{},{} }]:div(std_scale2[i])
   local rgb = images_test[i]
  local yuv = image.rgb2yuv(rgb)
  yuv[{ i,{},{} }]:add(-mean_scale1[i])
  yuv[{ i,{},{} }]:div(std_scale1[i])
  image.save(paths.concat(opt.save, d)..'test.jpg', yuv) 
  d = d+1
end

for i=1, #images_test do
  image.save(paths.concat(opt.save, i)..'test_preprocess1.jpg', scale_yuvs_test1[i][1])
  image.save(paths.concat(opt.save, i)..'test_preprocess2.jpg', scale_yuvs_test2[i][1])
end
  
-- create dataset
count = 1
count_class1 = 0
count_class2 = 0
count_class3 = 0
count_class4 = 0
for i = 1,#seg_images_test do 
  local modImg = seg_images_test[i]
  local scale_seg_image = image.scale(seg_images_test[i], seg_images_test[i]:size()[3]/2, seg_images_test[i]:size()[2]/2)
  for h=1,scale_seg_image:size()[2] do
    for w=1,scale_seg_image:size()[3]  do
     if (math.floor(seg_images_test[i][1][h][w]*2+0.5)>=1) and (count<=tesize) then
        testData.data_scale1[count][1]=h*2
        testData.data_scale1[count][2]=w*2
        testData.data_scale2[count][1]=h
        testData.data_scale2[count][2]=w
        testData.labels[count]=math.floor(scale_seg_image[1][h][w]*4+0.5)
        
        if testData.labels[count] == 1 then
          count_class1 = count_class1 + 1
          modImg[1][h][w] = 0
          modImg[2][h][w] = 0
          modImg[3][h][w] = 1
        elseif testData.labels[count] == 2 then
          count_class2 = count_class2 + 1
          modImg[1][h][w] = 0
          modImg[2][h][w] = 1
          modImg[3][h][w] = 0
        elseif testData.labels[count] == 3 then
          count_class3 = count_class3 + 1
          modImg[1][h][w] = 1
          modImg[2][h][w] = 0
          modImg[3][h][w] = 0
        else
          count_class4 = count_class4 + 1
          modImg[1][h][w] = 1
          modImg[2][h][w] = 1
          modImg[3][h][w] = 1
        end
        
        
        
        testData.link[count] = i
        count = count + 1
      end
    end
  end 
  image.save(i .. 'ModifiedTestImage.jpg', modImg)
end
print ('count test ', count)
print ('Class1: ', count_class1, ' class2: ', count_class2, ' class3: ', count_class3, ' class4: ', count_class4)
if count<=trsize then
  trsize = count-1
end
----------------------------------------------------------------------
-- define training and testing functions
--

-- this matrix records the current confusion across classes
confusion = optim.ConfusionMatrix(classes)

-- log results to files
accLogger = optim.Logger(paths.concat(opt.save, 'accuracy.log'))
errLogger = optim.Logger(paths.concat(opt.save, 'error.log'   ))

-- display function
function display(input)
   iter = iter or 0
   require 'image'
   win_input = image.display{image=input, win=win_input, zoom=2, legend='input'}
   if iter % 10 == 0 then
      if opt.model == 'convnet' then
         win_w1 = image.display{
            image=model:get(1).weight, zoom=4, nrow=10,
            min=-1, max=1,
            win=win_w1, legend='stage 1: weights', padding=1
         }
         win_w2 = image.display{
            image=model:get(4).weight, zoom=4, nrow=30,
            min=-1, max=1,
            win=win_w2, legend='stage 2: weights', padding=1
         }
      elseif opt.model == 'mlp' then
         local W1 = torch.Tensor(model:get(2).weight):resize(2048,1024)
         win_w1 = image.display{
            image=W1, zoom=0.5, min=-1, max=1,
            win=win_w1, legend='W1 weights'
         }
         local W2 = torch.Tensor(model:get(2).weight):resize(10,2048)
         win_w2 = image.display{
            image=W2, zoom=0.5, min=-1, max=1,
            win=win_w2, legend='W2 weights'
         }
      end
   end
   iter = iter + 1
end

function getPatch(image_in, i, j)
  return image.crop(image_in, j-1-pwidth/2, i-1-pheigh/2,j-1+pwidth/2,i-1+pheigh/2)
end
-- training function
function  train(dataset)
   -- epoch tracker
   epoch = epoch or 1
   
   -- local vars
   local time = sys.clock()
   local trainError = 0

   -- do one epoch
   print('<trainer> on training set:')
   print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')
   
   
   local shuffle = torch.randperm(math.min(dataset.data_scale1:size()[1],trsize))
   for t = 1,math.min(dataset.data_scale1:size()[1],trsize),opt.batchSize do
      -- disp progress
      --xlua.progress(t, dataset:size())

      -- create mini batch
      if (math.min(dataset.data_scale1:size()[1],trsize)-t>=opt.batchSize) then
        inputs_scale1 = torch.Tensor(opt.batchSize, 16,pwidth, pheigh)
        inputs_scale2 = torch.Tensor(opt.batchSize, 16,pwidth, pheigh)
        targets = torch.Tensor(opt.batchSize)
        numPatch = opt.batchSize
      else 
        numPatch = math.min(dataset.data_scale1:size()[1],trsize)-t+1
        inputs_scale1 = torch.Tensor(numPatch, 16,pwidth, pheigh)
        inputs_scale2 = torch.Tensor(numPatch, 16,pwidth, pheigh)
        targets = torch.Tensor(numPatch)
      end
            
      for i = t,math.min(t+opt.batchSize-1,math.min(dataset.data_scale1:size()[1],trsize)) do
         -- load new sample
         local input = getPatch(scale_yuvs_train1[dataset.link[shuffle[i]]],dataset.data_scale1[shuffle[i]][1],dataset.data_scale1[shuffle[i]][2])
         inputs_scale1[i-t+1] = convert3to16channels(input)
         input = getPatch(scale_yuvs_train2[dataset.link[shuffle[i]]],dataset.data_scale2[shuffle[i]][1],dataset.data_scale2[shuffle[i]][2])
         inputs_scale2[i-t+1] = convert3to16channels(input)
         targets[i-t+1] = dataset.labels[shuffle[i]]
      end

      -- create closure to evaluate f(X) and df/dX
      local feval = function(x)
         -- get new parameters
         if x ~= parameters then
            parameters:copy(x)
         end

         -- reset gradients
         gradParameters:zero()

         -- f is the average of all criterions
         local f1 = 0
         local f2 = 0

         -- evaluate function for complete mini batch
         -- estimate f
          local output1 = model.modules[1]:forward(inputs_scale1:cuda())
          local err1 = criterion1:forward(output1, targets)
          f1 = f1 + err1
          -- estimate df/dW
          local df_do1 = criterion1:backward(output1, targets)
          model.modules[1]:backward(inputs_scale1:cuda(), df_do1)
          
          -- estimate f
          local output2 = model.modules[2]:forward(inputs_scale2:cuda())
          local err2 = criterion2:forward(output2, targets)
          f2 = f2 + err2
          -- estimate df/dW
          local df_do2 = criterion2:backward(output2, targets)
          model.modules[2]:backward(inputs_scale2:cuda(), df_do2)

          -- update confusion
          for k = 1,numPatch do
            confusion:add(output1[k], targets[k])
            confusion:add(output2[k], targets[k])
          end

          -- visualize?
          if opt.visualize then
             display(inputs_scale1)
          end

         -- normalize gradients and f(X)
         gradParameters:div(2*inputs_scale1:size()[1])
         trainError = trainError + (f1 + f2)/(2*targets:size()[1])

         -- return f and df/dX
         return trainError,gradParameters
      end

      -- optimize on current mini-batchw
      if opt.optimization == 'CG' then
         config = config or {maxIter = opt.maxIter}
         optim.cg(feval, parameters, config)

      elseif opt.optimization == 'LBFGS' then
         config = config or {learningRate = opt.learningRate,
                             maxIter = opt.maxIter,
                             nCorrection = 10}
         optim.lbfgs(feval, parameters, config)

      elseif opt.optimization == 'SGD' then
         config = config or {learningRate = opt.learningRate,
                             maxIter = opt.maxIter,
                             weightDecay = opt.weightDecay,
                             momentum = opt.momentum,
                             learningRateDecay = 5e-5}
         optim.sgd(feval, parameters, config)

      elseif opt.optimization == 'ASGD' then
         config = config or {eta0 = opt.learningRate,
                             t0 = nbTrainingPatches * opt.t0}
         _,_,average = optim.asgd(feval, parameters, config)

      else
         error('unknown optimization method')
      end
   end

   -- train error
   trainError = trainError / math.floor(dataset:size()/opt.batchSize)
   torch.save('train error', trainError)

   -- time taken
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

   -- print confusion matrix
   print(confusion)
   confusion:updateValids()
   local trainAccuracy = confusion.averageValid * 100
   confusion:zero()

   -- save/log current net
   local filename = paths.concat(opt.save, 'NN.net')
   os.execute('mkdir -p ' .. paths.dirname(filename))
   if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
   end
   print('<trainer> saving network to '..filename)
   torch.save(filename, model)

   -- next epoch
   epoch = epoch + 1

   return trainAccuracy, trainError
end

-- test function
function test(dataset)
   -- local vars
   local testError = 0
   local time = sys.clock()

   -- averaged param use?
   if average then
      cachedparams = parameters:clone()
      parameters:copy(average)
   end

   -- test over given dataset
   print('<trainer> on testing Set:')
   for t = 1,math.min(dataset.data_scale1:size()[1],tesize), opt.batchSize do
      -- disp progress
      --xlua.progress(t, math.min(dataset.data_scale1:size()[1],tesize))
      
      -- create mini batch
      if (math.min(dataset.data_scale1:size()[1],trsize)-t>=opt.batchSize) then
        inputs_scale1 = torch.Tensor(opt.batchSize, 16,pwidth, pheigh)
        inputs_scale2 = torch.Tensor(opt.batchSize, 16,pwidth, pheigh)
        targets = torch.Tensor(opt.batchSize)
        numPatch = opt.batchSize
      else         
        numPatch = math.min(dataset.data_scale1:size()[1],trsize)-t+1
        inputs_scale1 = torch.Tensor(numPatch, 16,pwidth, pheigh)
        inputs_scale2 = torch.Tensor(numPatch, 16,pwidth, pheigh)
        targets = torch.Tensor(numPatch)
      end
      
      for i = t,math.min(t+opt.batchSize-1,math.min(dataset.data_scale1:size()[1],trsize)) do
         -- load new sample
         local input = getPatch(scale_yuvs_test1[dataset.link[i]],dataset.data_scale1[i][1],dataset.data_scale1[i][2])
         inputs_scale1[i-t+1] = convert3to16channels(input)
         input = getPatch(scale_yuvs_test2[dataset.link[i]],dataset.data_scale2[i][1],dataset.data_scale2[i][2])
         inputs_scale2[i-t+1] = convert3to16channels(input)
         targets[i-t+1] = dataset.labels[i]
      end
      
      -- test sample 
      local output1 = model.modules[1]:forward(inputs_scale1:cuda())
      local err1 = criterion1:forward(output1, targets)
      local output2 = model.modules[2]:forward(inputs_scale2:cuda())
      local err2 = criterion2:forward(output2, targets)
      
      -- update confusion
      for k = 1,numPatch do
        confusion:add(output1[k], targets[k])
        confusion:add(output2[k], targets[k])
      end
      
      -- compute error
      testError = testError + err1 + err2
   end

   -- timing
   time = sys.clock() - time
   time = time / dataset:size()
   print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

   -- testing error estimation
   testError = testError / (2*math.min(dataset.data_scale1:size()[1],tesize))

   -- print confusion matrix
   print(confusion)
   confusion:updateValids()
   local testAccuracy = confusion.averageValid * 100
   confusion:zero()

   -- averaged param use?
   if average then
      -- restore parameters
      parameters:copy(cachedparams)
   end

   return testAccuracy, testError
end

----------------------------------------------------------------------
-- and train!
--
max_acc = 0
while true do
   -- train/test
   trainAcc, trainErr = train(trainData)
   testAcc,  testErr  = test (testData)
   
   -- visual the train, test images
   for i=1, scale_yuvs_test1:size()[1] do
    image.save(i .. i .. '.jpg', scale_yuvs_test1[i])
    image.save(i .. i .. 'seg.jpg', seg_images_test[i])
   end
   for i=1, scale_yuvs_train1:size()[1] do
    image.save(i .. i .. i .. '.jpg', scale_yuvs_train1[i])
    image.save(i .. i .. i .. 'seg.jpg', seg_images[i])
   end
   
   if (testAcc>max_acc) then
     max_acc = testAcc
     -- save/log current net
     local filename = paths.concat(opt.save, 'Max_NN.net')
     os.execute('mkdir -p ' .. paths.dirname(filename))
     if paths.filep(filename) then
        os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
     end
     print('<trainer> saving network to '..filename)
     torch.save(filename, model)
    end 

   -- update logger
   accLogger:add{['% train accuracy'] = trainAcc, ['% test accuracy'] = testAcc}
   errLogger:add{['% train error']    = trainErr, ['% test error']    = testErr}

   -- plot logger
   accLogger:style{['% train accuracy'] = '-', ['% test accuracy'] = '-'}
   errLogger:style{['% train error']    = '-', ['% test error']    = '-'}
   accLogger:plot()
   errLogger:plot()
end