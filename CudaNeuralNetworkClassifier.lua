require 'nn'
require 'optim'
require 'image'
require 'torch'
require 'xlua'
require 'cutorch'
require 'cunn'
require 'cudnn'
require 'paths'

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
cmd:option('-learningRate', 1e-3, 'learning rate at t=0')
cmd:option('-batchSize', 100, 'mini-batch size (1 = pure stochastic)')
cmd:option('-weightDecay', 0, 'weight decay (SGD only)')
cmd:option('-momentum', 0, 'momentum (SGD only)')
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
-- train, test folder directories
dir = '/opt/zbstudio/myprograms/FeatureMaps/Fold1/Train'
dir_test = '/opt/zbstudio/myprograms/FeatureMaps/Fold1/Test'
classes = {1, 2, 3, 4}
trsize = 3500000
tesize = 100000
twidth = 32
theigh = 32
pwidth = 46 -- width of a patch centered a pixel
pheigh = 46 -- heigh of a patch centered a pixel
iwidth = 2250 --2250 -- width of input image 
iheigh = 1750 --1750 -- heigh of input image
nimage = 2 -- the number of input

-- load dataset
count_class1 = 320662
count_class2 = 240799
count_class3 = 108165
count_class4 = 126862

count = 1

-- define model to train
local total = count_class1 + count_class2 + count_class3 + count_class4
weight = torch.Tensor{total/count_class1, total/count_class2, total/count_class3, total/count_class4}
print('weight ', weight)

mlp = nn.Sequential();  -- make a multi-layer perceptron
mlp:add(nn.Linear(200*2, 2048):cuda())
mlp:add(nn.ReLu())
mlp:add(nn.Linear(2048, 4096):cuda())
mlp:add(nn.ReLu())
mlp:add(nn.Linear(4096, #classes):cuda())
mlp:add(nn.LogSoftMax())
mlp = mlp:cuda()
cudnn.convert(mlp, cudnn)

criterion = nn.CrossEntropyCriterion(weight):cuda()

-- retrieve parameters and gradients
parameters,gradParameters = mlp:getParameters()

-- verbose
print('using model:')
print(mlp)
------------------------------------------------------------
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
        image=mlp:get(1).weight, zoom=4, nrow=10,
        min=-1, max=1,
        win=win_w1, legend='stage 1: weights', padding=1
      }
      win_w2 = image.display{
        image=mlp:get(4).weight, zoom=4, nrow=30,
        min=-1, max=1,
        win=win_w2, legend='stage 2: weights', padding=1
      }
    elseif opt.model == 'mlp' then
      local W1 = torch.Tensor(mlp:get(2).weight):resize(2048,1024)
      win_w1 = image.display{
        image=W1, zoom=0.5, min=-1, max=1,
        win=win_w1, legend='W1 weights'
      }
      local W2 = torch.Tensor(mlp:get(2).weight):resize(10,2048)
      win_w2 = image.display{
        image=W2, zoom=0.5, min=-1, max=1,
        win=win_w2, legend='W2 weights'
      }
    end
  end
  iter = iter + 1
end

print_confusion = 0

-- training function
function  train(dataset, count)
  -- epoch tracker
  count_tr = 0
  epoch = epoch or 1

  -- local vars
  local time = sys.clock()
  local trainError = 0

  -- do one epoch
  print('<trainer> on training set:')
  print("<trainer> online epoch # " .. epoch .. ' [batchSize = ' .. opt.batchSize .. ']')

  local shuffle = torch.randperm(count)
  for t = 1,count, opt.batchSize do
    -- disp progress
    xlua.progress(t, count)

    -- create mini batch
    local inputs = {}
    local targets = {}
    for i = t,math.min(t+opt.batchSize-1, count) do
      -- load new sample
      local input = dataset.data[shuffle[i]]
      table.insert(inputs, input)
      
      local target = dataset.labels[shuffle[i]]
      table.insert(targets, target)
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
      local f = 0
      count_tr = count_tr+ 1

      -- evaluate function for complete mini batch
      for i = 1,#targets do
        -- estimate f
        local output = mlp:forward(inputs[i]:cuda())
        local err = criterion:forward(output, targets[i])
        f = f + err
        -- estimate df/dW
        local df_do = criterion:backward(output, targets[i])
          mlp:backward(inputs[i]:cuda(), df_do)

        -- update confusion
        confusion:add(output, targets[i])

        -- visualize?
        if opt.visualize then
          display(inputs[i])
        end
      end

      -- normalize gradients and f(X)
      gradParameters:div(#inputs)
      trainError = trainError + f/(#inputs)

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
        learningRateDecay = 5e-7}
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
    trainError = trainError / math.floor(count/opt.batchSize)

    -- time taken
    time = sys.clock() - time
    time = time / dataset:size()
    print("<trainer> time to learn 1 sample = " .. (time*1000) .. 'ms')

    -- print confusion matrix
    print(confusion)
    local trainAccuracy = confusion.totalValid * 100
    confusion:zero()

    -- save/log current net
    local filename = paths.concat(opt.save, 'Cuda_NNClassifier.net')
    os.execute('mkdir -p ' .. paths.dirname(filename))
    if paths.filep(filename) then
      os.execute('mv ' .. filename .. ' ' .. filename .. '.old')
    end
    print('<trainer> saving network to '..filename)
    torch.save(filename, mlp)

    -- next epoch
    epoch = epoch + 1
    return trainAccuracy, trainError
  end

-- test function
function test(dataset, count)
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
  for t = 1,math.min(dataset.data:size()[1],count) do
    -- disp progress
    --xlua.progress(t, math.min(dataset.data:size()[1],tesize))
    -- get new sample
    local input = torch.Tensor(256*3)
    input = dataset.data[t]
    local target = dataset.labels[t]

    -- test sample 
    local output = mlp:forward(input:cuda())
    local err = criterion:forward(output, target)

    -- update confusion
    confusion:add(output, target)

    -- compute error
    testError = testError + err
  end

  -- timing
  time = sys.clock() - time
  time = time / dataset:size()
  print("<trainer> time to test 1 sample = " .. (time*1000) .. 'ms')

  -- testing error estimation
  testError = testError / dataset:size()

  -- print confusion matrix
  print(confusion)
  local testAccuracy = confusion.totalValid * 100
  confusion:zero()

  -- averaged param use?p
  if average then
    -- restore parameters
    parameters:copy(cachedparams)
  end

  return testAccuracy, testError
end

-- load dataset
testData = {
  data = torch.Tensor(tesize, 200*2):cuda(),
  pos = torch.IntTensor(tesize, 2):cuda(),
  link = torch.IntTensor(tesize):cuda(),
  labels = torch.IntTensor(tesize):cuda(),
  size = function() return tesize end
}
-- load test dataset
-- load data input
count_test = 1
count_class1 = 0
count_class2 = 0
count_class3 = 0
count_class4 = 0
local list1_tensor = torch.load(dir_test .. '/1list1')
local list2_tensor = torch.load(dir_test .. '/1list2'
local map1_tensor = torch.load(dir_test .. '/1maps1')
local map2_tensor = torch.load(dir_test .. '/1maps2')
if paths.dirp(dir_test .. '/1seg.jpg') then
  label = image.load(dir_test .. '/1seg.jpg')
else
  label = image.load(dir_test .. '/1seg.png')
end
image.save('1test_seg.jpg', label)
last_label_map2 = 1
last_label_map3 = 1
for h=1, label:size(2) do
  for w=1, label:size(3) do
    if math.floor(label[1][h][w]*4+0.5)>=1 and map1_tensor[h][w]~=0 and count_test<=tesize then
      if map2_tensor[h][w]~=0 then
        last_label_map2 = map2_tensor[h][w]
      else
        map2_tensor[h][w] = last_label_map2
      end
      local vt = torch.cat(list1_tensor[map1_tensor[h][w]], list2_tensor[map2_tensor[h][w]], 1)
      if count_test<=tesize then
        testData.data[count_test] = vt
        testData.pos[count_test][1] = h
        testData.pos[count_test][2] = w
        testData.labels[count_test] = math.floor(label[1][h][w]*4+0.5) 
        if testData.labels[count_test] == 1 then
          count_class1 = count_class1 + 1
        elseif testData.labels[count_test] == 2 then
          count_class2 = count_class2 + 1
        elseif testData.labels[count_test] == 3 then
          count_class3 = count_class3 + 1
        else
          count_class4 = count_class4 + 1
        end
        testData.link = i
        count_test = count_test + 1
      end
    end
  end 
end
print ('Test Dataset --> Class1: ', count_class1, ' class2: ', count_class2, ' class3: ', count_class3, ' class4: ', count_class4)
print('Loaded dataset for testing successful!')
max_acc = 0

trainData = {
  data = torch.Tensor(trsize, 200*2),
  pos = torch.IntTensor(trsize, 2),
  link = torch.IntTensor(trsize),
  labels = torch.IntTensor(trsize),
  size = function() return trsize end
}

while true do
  -- load train dataset
  trainAccTotal = 0
  trainErrTotal = 0
  for i=1, 6 do
    -- load data input
    count_class1 = 0
    count_class2 = 0
    count_class3 = 0
    count_class4 = 0
    count = 1
    print('--> loading picture ' .. i)
    local list1_tensor = torch.load(dir .. '/' .. i .. 'list1')
    local list2_tensor = torch.load(dir .. '/' .. i .. 'list2')
    local map1_tensor = torch.load(dir .. '/' .. i .. 'maps1')
    local map2_tensor = torch.load(dir .. '/' .. i .. 'maps2')      
    if paths.dirp(dir .. '/' .. i .. 'seg.png') then
      label = image.load(dir .. '/' .. i .. 'seg.png')
    else
      label = image.load(dir .. '/' .. i .. 'seg.jpg')
    end
    image.save(i .. 'train_seg.jpg', label)
    for h=1, label:size()[2] do
      for w=1, label:size()[3] do
        if  count <= trsize  and  math.floor(label[1][h][w]*4+0.5)>=1 and map1_tensor[h][w]~=0 and map2_tensor[h][w]~=0 and map3_tensor[h][w]~=0 then
          local vt = torch.cat(list1_tensor[map1_tensor[h][w]], list2_tensor[map2_tensor[h][w]], 1):cat(list3_tensor[map3_tensor[h][w]], 1)
          trainData.data[count] = nn.utils.recursiveType(vt, 'torch.DoubleTensor')
          trainData.pos[count][1] = h
          trainData.pos[count][2] = w
          trainData.labels[count] = math.floor(label[1][h][w]*4+0.5) 
          trainData.link = i     
          if trainData.labels[count] == 1 then
            count_class1 = count_class1 + 1
          elseif trainData.labels[count] == 2 then
            count_class2 = count_class2 + 1
          elseif trainData.labels[count] == 3 then
            count_class3 = count_class3 + 1
          else
            count_class4 = count_class4 + 1
          end
          count = count + 1
        end
      end 
    end
    print ('Train Dataset --> Class1: ', count_class1, ' class2: ', count_class2, ' class3: ', count_class3, ' class4: ', count_class4)
    trainAcc, trainErr = train(trainData, count)
    trainAccTotal = trainAccTotal + trainAcc
    trainErrTotal = trainErrTotal + trainErr
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