import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models

import network
from network import Conv2d, FC
import torchvision.models as models
import torch.utils.model_zoo as model_zoo
import math
import h5py

import argparse

import pdb

anchor_scales_normal = [2, 4, 8, 16, 32, 64]
anchor_ratios_normal = [0.25, 0.5, 1, 2, 4]

parser = argparse.ArgumentParser('Options for training RPN in pytorch')


def nms_detections(pred_boxes, scores, nms_thresh, inds=None):
    dets = np.hstack((pred_boxes,
                      scores[:, np.newaxis])).astype(np.float32)
    keep = nms(dets, nms_thresh)
    if inds is None:
        return pred_boxes[keep], scores[keep]
    return pred_boxes[keep], scores[keep], inds[keep]


class LungDataset(Dataset):

	def __init__(self, path_to_dataset, only_positive):
	self.path = path_to_dataset
	self.dataset = pandas.read_hdf(self.path_to_dataset).as_matrix()

	if only_positive:
	dataset = dataset[dataset[:, 5] == 1]

	def __getitem__(self, index):
		return 

	def __len__(self):
        return int(len(self.dict_of_gpu_index_to_list_of_video_indices[0]))

class LungObjectLocalizer(nn.module):

	anchor_scales_kmeans = [19.944, 9.118, 35.648, 42.102, 23.476, 15.882, 6.169, 9.702, 6.072, 32.254, 3.294, 10.148, 22.443, \
							13.831, 16.250, 27.969, 14.181, 27.818, 34.146, 29.812, 14.219, 22.309, 20.360, 24.025, 40.593, ]
	anchor_ratios_kmeans = [2.631, 2.304, 0.935, 0.654, 0.173, 0.720, 0.553, 0.374, 1.565, 0.463, 0.985, 0.914, 0.734, 2.671, \
							0.209, 1.318, 1.285, 2.717, 0.369, 0.718, 0.319, 0.218, 1.319, 0.442, 1.437, ]

	def _init_(input_dim, output_dim):
		self.input_dim
		self.output_dim
		self.anchor_scales = self.anchor_scales_kmeans
		self.anchor_ratios = self.anchor_ratios_kmeans
		self.anchor_num = len(self.anchor_scales)

		self.features = models.vgg16(pretrained=True).features
		self.features.__delattr__('30')
		network.set_trainable_param(list(self.features.parameters())[:8], requires_grad=False)
		self.conv1 = Conv2d(512, 512, 3, same_padding=True)
		self.score_conv = Conv2d(512, self.anchor_num * 2, 1, relu=False, same_padding=False)
		self.bbox_conv = Conv2d(512, self.anchor_num * 4, 1, relu=False, same_padding=False)
		# loss
        self.cross_entropy = None
        self.loss_box = None

        # initialize the parameters
        self.initialize_parameters()


	def initialize_parameters(self, normal_method='normal'):
		normal_fun = network.weights_normal_init
		normal_fun(self.conv1, 0.025)
		normal_fun(self.score_conv, 0.025)
		normal_fun(self.bbox_conv, 0.01)

	def _unmap(data, count, inds, fill=0):
		""" Unmap a subset of item (data) back to the original set of items (of
		size count) """
		if len(data.shape) == 1:
			ret = np.empty((count,), dtype=np.float32)
			ret.fill(fill)
			ret[inds] = data
		else:
			ret = np.empty((count,) + data.shape[1:], dtype=np.float32)
			ret.fill(fill)
			ret[inds, :] = data
		return ret


	def _compute_targets(ex_rois, gt_rois):
		"""Compute bounding-box regression targets for an image."""

		assert ex_rois.shape[0] == gt_rois.shape[0]
		assert ex_rois.shape[1] == 4
		assert gt_rois.shape[1] >= 4

		targets = bbox_transform(ex_rois, gt_rois[:, :4]).astype(np.float32, copy=False)

	def reshape_layer(x, d):
		input_shape = x.size()
		x = x.view(
			input_shape[0],
			int(d),
			int(float(input_shape[1] * input_shape[2]) / float(d)),
			input_shape[3])
        return x


	def anchor_target_layer(rpn_cls_score, gt_boxes, im_info, anchor_scales, anchor_ratios, _feat_stride=[16, ]):
		"""
		Assign anchors to ground-truth targets. Produces anchor classification
		labels and bounding-box regression targets.
		----------
		rpn_cls_score: for pytorch (1, Ax2, H, W) bg/fg scores of previous conv layer
		gt_boxes: (G, 4) vstack of [x1, y1, x2, y2]
		im_info: a list of [image_height, image_width, scale_ratios]
		_feat_stride: the downsampling ratio of feature map to the original input image
		anchor_scales: the scales to the basic_anchor (basic anchor is [16, 16])
		----------
		Returns
		----------
		rpn_labels : (HxWxA, 1), for each anchor, 0 denotes bg, 1 fg, -1 dontcare
		rpn_bbox_targets: (HxWxA, 4), distances of the anchors to the gt_boxes(may contains some transform)
		                        that are the regression objectives
		rpn_bbox_inside_weights: (HxWxA, 4) weights of each boxes, mainly accepts hyper param in cfg
		rpn_bbox_outside_weights: (HxWxA, 4) used to balance the fg/bg,
		                        beacuse the numbers of bgs and fgs mays significiantly different
		"""
		_anchors = generate_anchors(scales=anchor_scales, ratios=anchor_ratios) # 9 anchor boxes (with respect to the original image)
		_num_anchors = _anchors.shape[0]

		# allow boxes to sit over the edge by a small amount
		_allowed_border = cfg.TRAIN.RPN_ALLOWED_BORDER

		im_info = im_info[0]

		# Algorithm:
		#
		# for each (H, W) location i from feature map
		#	1.) generate 9 anchor boxes centered on cell i
		#   2.) find the anchor boxes that are inside the cell
		# filter out-of-image anchors
		# measure GT overlap

		assert rpn_cls_score.shape[0] == 1, \
		'Only single item batches are supported'

		# map of shape (..., H, W)
		# pytorch (bs, c, h, w)
		height, width = rpn_cls_score.shape[2:4]


		# 1. Generate proposals from shifted anchors
		shift_x = np.arange(0, width) * _feat_stride
		shift_y = np.arange(0, height) * _feat_stride
		shift_x, shift_y = np.meshgrid(shift_x, shift_y)
		shifts = np.vstack((shift_x.ravel(), shift_y.ravel(), shift_x.ravel(), shift_y.ravel())).transpose()
		
		# add A anchors (1, A, 4) to
		# cell K shifts (K, 1, 4) to get
		# shift anchors (K, A, 4)
		# reshape to (K*A, 4) shifted anchors
		A = _num_anchors
		K = shifts.shape[0]
		all_anchors = (_anchors.reshape((1, A, 4)) + shifts.reshape((1, K, 4)).transpose((1, 0, 2)))
		all_anchors = all_anchors.reshape((K * A, 4))
		total_anchors = int(K * A)

		# only keep anchors inside the image
		inds_inside = np.where(
		    (all_anchors[:, 0] >= -_allowed_border) &
		    (all_anchors[:, 1] >= -_allowed_border) &
		    (all_anchors[:, 2] < im_info[1] + _allowed_border) &  # width
		    (all_anchors[:, 3] < im_info[0] + _allowed_border)  # height
		)[0]

	    # keep only inside anchors
	    anchors = all_anchors[inds_inside, :]
	    # label: 1 is positive, 0 is negative, -1 is dont care
	    labels = np.empty((len(inds_inside),), dtype=np.float32)
	    labels.fill(-1)

		# overlaps between the anchors and the gt boxes
		# overlaps (ex, gt)
		overlaps = bbox_overlaps(np.ascontiguousarray(anchors, dtype=np.float), np.ascontiguousarray(gt_boxes, dtype=np.float)) # K*A, G
		argmax_overlaps = overlaps.argmax(axis=1)
		max_overlaps = overlaps[np.arange(len(inds_inside)), argmax_overlaps] # (K*A,) for each K*A_i, which ground truth created the most overlap with K*A_i
		gt_argmax_overlaps = overlaps.argmax(axis=0)
		gt_max_overlaps = overlaps[gt_argmax_overlaps, np.arange(overlaps.shape[1])] # (G,) for each G_i, which KA created the most overlap with G_i 
		gt_argmax_overlaps = np.where(overlaps == gt_max_overlaps)[0] # finds the rows  in which that contains at least one maximum for a specific G ground truth, assigns index of the row to the corresponding ground truth

		# fg label: for each gt, anchor with highest overlap
		# need to generate as much proposals that gives us the most number of bounding boxes that gives us foreground and as high of an overlap
		labels[gt_argmax_overlaps] = 1

		
		bbox_targets = _compute_targets(anchors, gt_boxes[argmax_overlaps, :])


		bbox_inside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
		bbox_inside_weights[labels == 1, :] = np.array(cfg.TRAIN.RPN_BBOX_INSIDE_WEIGHTS)

		bbox_outside_weights = np.zeros((len(inds_inside), 4), dtype=np.float32)
		if cfg.TRAIN.RPN_POSITIVE_WEIGHT < 0:
			# uniform weighting of examples (given non-uniform sampling)
			num_examples = np.sum(labels >= 0)
			positive_weights = np.ones((1, 4)) * 1.0 / num_examples
			negative_weights = np.ones((1, 4)) * 1.0 / num_examples
		else:
			assert ((cfg.TRAIN.RPN_POSITIVE_WEIGHT > 0) &
			        (cfg.TRAIN.RPN_POSITIVE_WEIGHT < 1))
			positive_weights = (cfg.TRAIN.RPN_POSITIVE_WEIGHT /
			                    np.sum(labels == 1))
			negative_weights = ((1.0 - cfg.TRAIN.RPN_POSITIVE_WEIGHT) /
			                    np.sum(labels == 0))
		bbox_outside_weights[labels == 1, :] = positive_weights
		bbox_outside_weights[labels == 0, :] = negative_weights



		# map up to original set of anchors
		labels = _unmap(labels, total_anchors, inds_inside, fill=-1)
		bbox_targets = _unmap(bbox_targets, total_anchors, inds_inside, fill=0)
		bbox_inside_weights = _unmap(bbox_inside_weights, total_anchors, inds_inside, fill=0)
		bbox_outside_weights = _unmap(bbox_outside_weights, total_anchors, inds_inside, fill=0)

		labels = labels.reshape((1, height, width, A)).transpose(0, 3, 1, 2)
		labels = labels.reshape((1, 1, A * height, width))
		rpn_labels = labels.transpose(0, 2, 3, 1).reshape(-1)

		# bbox_targets
		bbox_targets = bbox_targets.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)

		rpn_bbox_targets = bbox_targets
		bbox_inside_weights = bbox_inside_weights.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
		rpn_bbox_inside_weights = bbox_inside_weights
		bbox_outside_weights = bbox_outside_weights.reshape((1, height, width, A * 4)).transpose(0, 3, 1, 2)
		rpn_bbox_outside_weights = bbox_outside_weights

		return rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights

	def build_loss(self, rpn_cls_score_reshape, rpn_bbox_pred, rpn_data):
		# classification loss
		rpn_cls_score = rpn_cls_score_reshape.permute(0, 2, 3, 1).contiguous().view(-1, 2)
		rpn_label = rpn_data[0]

		rpn_keep = Variable(rpn_label.data.ne(-1).nonzero().squeeze()).cuda()
		rpn_cls_score = torch.index_select(rpn_cls_score, 0, rpn_keep)
		rpn_label = torch.index_select(rpn_label, 0, rpn_keep)

		fg_cnt = torch.sum(rpn_label.data.ne(0))
		bg_cnt = rpn_label.data.numel() - fg_cnt
		_, predict = torch.max(rpn_cls_score.data, 1)
		error = torch.sum(torch.abs(predict - rpn_label.data))

	    self.tp = torch.sum(predict[:fg_cnt].eq(rpn_label.data[:fg_cnt]))
	    self.tf = torch.sum(predict[fg_cnt:].eq(rpn_label.data[fg_cnt:]))
	    self.fg_cnt = fg_cnt
	    self.bg_cnt = bg_cnt

	    # Build a detector that determines if a bounding box is an over an object
	    rpn_cross_entropy = F.cross_entropy(rpn_cls_score, rpn_label)

		# Only on those boxes that have been identified as having a good overlap over the object we are detecting, 
		# we want to gauge how far off these predictions and update parameters to make better predictions that will lead to higher degrees of overalaps
		rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = rpn_data[1:]
		rpn_bbox_targets = torch.mul(rpn_bbox_targets, rpn_bbox_inside_weights)
		rpn_bbox_pred = torch.mul(rpn_bbox_pred, rpn_bbox_inside_weights)
		rpn_loss_box = F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, size_average=False) /  (fg_cnt + 1e-4)

		return rpn_cross_entropy, rpn_loss_box


        # print 'Smooth L1 loss: ', F.smooth_l1_loss(rpn_bbox_pred, rpn_bbox_targets, size_average=False)
        # print 'fg_cnt', fg_cnt

	def forward(self, im_data, im_info, gt_objects=None):

		im_data = im_data.cuda()
		features = self.features(im_data)
		# print 'features.std()', features.data.std()
		rpn_conv1 = self.conv1(features)
		# print 'rpn_conv1.std()', rpn_conv1.data.std()
		# object proposal score
		rpn_cls_score = self.score_conv(rpn_conv1)
		# print 'rpn_cls_score.std()', rpn_cls_score.data.std()
		rpn_cls_score_reshape = self.reshape_layer(rpn_cls_score, 2)
		rpn_cls_prob = F.softmax(rpn_cls_score_reshape)
		rpn_cls_prob_reshape = self.reshape_layer(rpn_cls_prob, self.anchor_num*2)
		# rpn boxes
		rpn_bbox_pred = self.bbox_conv(rpn_conv1)
		# print 'rpn_bbox_pred.std()', rpn_bbox_pred.data.std() * 4

		# proposal layer
		cfg_key = 'TRAIN' if self.training else 'TEST'
		rois = self.proposal_layer(rpn_cls_prob_reshape, rpn_bbox_pred, im_info, cfg_key, self._feat_stride, self.anchor_scales, self.anchor_ratios)

		# generating training labels and build the rpn loss
		if self.training:
			rpn_data = self.anchor_target_layer(rpn_cls_score, gt_objects, im_info, self.anchor_scales, self.anchor_ratios, self._feat_stride)
			# Build a loss on the those proposals that we consider as having enough overlap to the object. With time, we hope to generate proposals that gives us the most overlap
			self.cross_entropy, self.loss_box = self.build_loss(rpn_cls_score_reshape, rpn_bbox_pred, rpn_data)

	    return features, rois


def train(train_loader, target_net, optimizer, epoch):
	batch_time = network.AverageMeter()
	data_time = network.AverageMeter()
	train_loss = network.AverageMeter()
	train_loss_obj_box = network.AverageMeter()
	train_loss_obj_entropy = network.AverageMeter()
	train_loss_reg_box = network.AverageMeter()
	train_loss_reg_entropy = network.AverageMeter()

	target_net.train()
	end = time.time()
	for i, (im_data, im_info, gt_objects) in enumerate(train_loader):
		# measure the data loading time
		data_time.update(time.time() - end)

		# Forward pass
		target_net(im_data, im_info.numpy(), gt_objects.numpy()[0])
		# record loss
		loss = target_net.loss
		# total loss
		train_loss.update(loss.data[0], im_data.size(0))
		# object bbox reg
		train_loss_obj_box.update(target_net.loss_box.data[0], im_data.size(0))
		# object score
		train_loss_obj_entropy.update(target_net.cross_entropy.data[0], im_data.size(0))

		# backward
		optimizer.zero_grad()
		loss.backward()
		if not args.disable_clip_gradient:
			network.clip_gradient(target_net, 10.)
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if  (i + 1) % args.log_interval == 0:
			print('Epoch: [{0}][{1}/{2}]\t'
				'Batch_Time: {batch_time.avg:.3f}s\t'
				'lr: {lr: f}\t'
				'Loss: {loss.avg:.4f}\n'
				'\t[object]: '
				'cls_loss: {cls_loss_object.avg:.3f}\t'
				'reg_loss: {reg_loss_object.avg:.3f}\n'.format(
					epoch, i + 1, len(train_loader), batch_time=batch_time,lr=args.lr,
					data_time=data_time, loss=train_loss,
					cls_loss_object=train_loss_obj_entropy, reg_loss_object=train_loss_obj_box))

def test(test_loader, target_net):
    box_num = 0
    correct_cnt, total_cnt = 0, 0
    print '========== Testing ======='
    target_net.eval()

    batch_time = network.AverageMeter()
    end = time.time()
    for i, (im_data, im_info, gt_objects) in enumerate(test_loader):
        correct_cnt_t, total_cnt_t = 0, 0
        # Forward pass
		object_rois = target_net(im_data, im_info.numpy(), gt_objects.numpy())[1:]
		box_num += object_rois.size(0)
		correct_cnt_t, total_cnt_t = check_recall(object_rois, gt_objects[0].numpy(), 50)
		correct_cnt += correct_cnt_t
		total_cnt += total_cnt_t
		batch_time.update(time.time() - end)
		end = time.time()
		if (i + 1) % 100 == 0 and i > 0:
			print('[{0}/{10}]  Time: {1:2.3f}s/img).'
					'\t[object] Avg: {2:2.2f} Boxes/im, Top-50 recall: {3:2.3f} ({4:d}/{5:d})'.format(
                  	i + 1, batch_time.avg,
					box_num[0] / float(i + 1), correct_cnt[0] / float(total_cnt[0])* 100, correct_cnt[0], total_cnt[0],
					len(test_loader)))

    recall = correct_cnt / total_cnt.astype(np.float)
    print '====== Done Testing ===='
    return recall

def main():
	
    train_set = visual_genome(args.dataset_option, 'train')
    test_set = visual_genome('small', 'test')
    print "Done."

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, shuffle=True, num_workers=8, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
    net = RPN(not args.use_normal_anchors)
    #if args.resume_training:
    #    print 'Resume training from: {}'.format(args.res ume_model)
    #    if len(args.resume_model) == 0:
    #        raise Exception('[resume_model] not specified')
    #    network.load_net(args.resume_model, net)
    #    optimizer = torch.optim.SGD([
    #            {'params': list(net.parameters())[26:]}, 
    #            ], lr=args.lr, momentum=args.momentum, weight_decay=0.0005)
	print 'Training from scratch...Initializing network...'
	optimizer = torch.optim.SGD(list(net.parameters())[26:], lr=args.lr, momentum=args.momentum, weight_decay=0.0005)

    network.set_trainable(net.features, requires_grad=False)
    #net.cuda()

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    best_recall = np.array([0.0, 0.0])

    for epoch in range(0, args.max_epoch):
        
        # Training
        train(train_loader, net, optimizer, epoch)

        # Testing
        recall = test(test_loader, net)
        print('Epoch[{epoch:d}]: '
              'Recall: '
              'object: {recall[0]: .3f}%% (Best: {best_recall[0]: .3f}%%)'
              'region: {recall[1]: .3f}%% (Best: {best_recall[1]: .3f}%%)'.format(
                epoch = epoch, recall=recall * 100, best_recall=best_recall * 100))
        # update learning rate
        if epoch % args.step_size == 0:
            args.disable_clip_gradient = True
            args.lr /= 10
            for param_group in optimizer.param_groups:
                param_group['lr'] = args.lr

        save_name = os.path.join(args.output_dir, '{}_epoch_{}.h5'.format(args.model_name, epoch))
        network.save_net(save_name, net)
        print('save model: {}'.format(save_name))

        if np.all(recall > best_recall):
            best_recall = recall
            save_name = os.path.join(args.output_dir, '{}_best.h5'.format(args.model_name, epoch))
            network.save_net(save_name, net)




