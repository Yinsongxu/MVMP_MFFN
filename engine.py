import torch
import torch.nn as nn
from loss import CrossEntropyLoss, SmoothLoss, TripletLoss,Ncut
from utils import AverageMeter, open_specified_layers, open_all_layers, save_checkpoint, mkdir_if_missing, accuracy, compute_distance_matrix, evaluate_rank, bregmance_divergence, euclidean_distance ,init_edge, cosine_distance, visualize_ranked_results
from torch.utils.tensorboard import SummaryWriter
import time
import datetime
from torch.nn import functional as F
import sys
import os
import os.path as osp
import numpy as np
import cv2
import gc
class Engine():
    def __init__(self, model, optimizer, datamanger, scheduler=None,
        use_gpu=True, label_smooth=True, writer=None, k = 9, gamma = 128):
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.use_gpu = (torch.cuda.is_available() and use_gpu)
        self.writer = writer
        self.trainloader, self.testloader = datamanger.return_dataloaders()
        self.datamanger = datamanger
        self.k=k
        self.cnn_criterion = CrossEntropyLoss(
            num_classes=datamanger.num_train_pids,
            use_gpu=self.use_gpu,
            label_smooth=label_smooth
        )
        
        self.gnn_criterion2 = TripletLoss()
        #self.gnn_criterion = SmoothLoss(k = k)
        #self.gnn_criterion = Ncut(gamma)
        
        if not isinstance(self.model, nn.Module):
            raise TypeError('model must be an instance of nn.Module')

    def run(self, save_dir='log', max_epoch=0, start_epoch=0, fixbase_epoch=0,
        open_layers=None, start_eval=0, eval_freq=-1, adjust_freq=10, test_only=False, print_freq=10, visrank = False,
        visrank_topk=10,use_metric_cuhk03=False, ranks=[1, 5, 10, 20], rerank=False):
        if test_only:
            self.test(0, self.testloader, save_dir=save_dir, visrank= visrank)
            return
        if self.writer is None:
            self.writer = SummaryWriter(log_dir=save_dir)
        time_start = time.time()
        print('=>Start training')
        for epoch in range(start_epoch, max_epoch):
            self.train(epoch, max_epoch, self.trainloader, fixbase_epoch, open_layers, print_freq)
            if (epoch+1)>=start_eval and eval_freq>0 and (epoch+1)%eval_freq==0 and (epoch+1)!=max_epoch and epoch>91:
                accuracy = self.test(epoch, self.testloader, save_dir = save_dir)
                #accuracy = 0
                self._save_checkpoint(epoch, accuracy, save_dir)
        if max_epoch>0:
            print('=> Final test')
            rank1 = self.test(epoch, self.testloader, save_dir=save_dir)
            #rank1=0
            self._save_checkpoint(epoch, rank1, save_dir)
        elapsed = round(time.time() - time_start)
        elapsed = str(datetime.timedelta(seconds=elapsed))
        print('Elapsed {}'.format(elapsed))
        if self.writer is not None:
            self.writer.close()


    def train(self, epoch, max_epoch, trainloader, fixbase_epoch=0, open_layers=None, print_freq=10):
        losses_cnn = AverageMeter()
        losses_gnn = AverageMeter()
        losses_tri = AverageMeter()
        losses_tri2 = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        self.model.train()
        if(epoch+1)<=fixbase_epoch and open_layers is not None:
            print('* Only train {} (epoch: {}/{})'.format(open_layers, epoch+1, fixbase_epoch))
            open_specified_layers(self.model, open_layers)
        else:
            open_all_layers(self.model)

        num_batches = len(trainloader)
        end = time.time()
        for batch_idx, data in enumerate(trainloader):
            data_time.update(time.time()-end)
            imgs, labels = self._parse_data(data)
            if self.use_gpu:
                imgs = imgs.cuda()
                labels = labels.cuda()
            self.optimizer.zero_grad()
            #classifier, affinity = self.model(imgs)
            gnn_cls, cnn_cls, feat, feat_cnn = self.model(imgs)
            gnn_loss = self.cnn_criterion(gnn_cls, labels)
            #cnn_loss = self.cnn_criterion(cnn_cls, labels)
            tri_gnn = self.gnn_criterion2(feat, labels) 
            #tri_cnn = self.gnn_criterion2(feat_cnn, labels) 
            loss =  gnn_loss + tri_gnn 
            loss.backward()
            self.optimizer.step()
            batch_time.update(time.time()-end)
            losses_cnn.update(gnn_loss.item(), labels.size(0))
            losses_gnn.update(gnn_loss.item(), labels.size(0))
            losses_tri.update(tri_gnn.item(), labels.size(0))
            accs.update(accuracy(gnn_cls, labels)[0].item())    

            if (batch_idx+1)%print_freq == 0:
                eta_seconds = batch_time.avg * (num_batches-(batch_idx+1) + (max_epoch-(epoch+1))*num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss_cnn {cnn_loss.val:.4f} ({cnn_loss.avg:.4f})\t'
                      'Loss_gnn {gnn_loss.val:.4f}({gnn_loss.avg:.4f})\t'
                      'Loss_tri {tri_loss.val:.4f}({tri_loss.avg:.4f})\t'
                      'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                      'Lr {lr:.6f}\t'
                      'eta {eta}'.format(
                       epoch+1, max_epoch, batch_idx+1, num_batches,
                       batch_time=batch_time,
                       data_time=data_time,
                       cnn_loss = losses_cnn,
                       gnn_loss = losses_gnn,
                       tri_loss = losses_tri,
                       acc=accs,
                       lr=self.optimizer.param_groups[0]['lr']   ,
                       eta=eta_str
                      )
                )
            if self.writer is not None:
                n_iter = epoch * num_batches + batch_idx
                self.writer.add_scalar('Train/Time', batch_time.avg, n_iter)
                self.writer.add_scalar('Train/Data', data_time.avg, n_iter)
                self.writer.add_scalar('Train/Loss_ces', losses_cnn.avg, n_iter)
                self.writer.add_scalar('Train/Loss_gnn', losses_gnn.avg, n_iter)
                self.writer.add_scalar('Train/Acc', accs.avg, n_iter)
                self.writer.add_scalar('Train/Lr', self.optimizer.param_groups[0]['lr'], n_iter)
            
            end = time.time()

        if self.scheduler is not None:
            self.scheduler.step()


    def test(self, epoch, testloader, dist_metric='euclidean', save_dir='', normalize_feature=False,
    visrank=False, visrank_topk=10, use_metric_cuhk03=False, ranks=[1,5,10,20]):
        print('=> Start test...')
        targets = list(testloader.keys())
        for name in targets:
            domain = 'source' if name in self.datamanger.sources else 'target'
            print('#### Evaluating {} ({}) ####'.format(name,domain))
            queryloader = testloader[name]['query']
            galleryloader = testloader[name]['gallery']
            rank1 = self._evaluate(
                epoch,
                dataset_name=name,
                queryloader=queryloader,
                galleryloader=galleryloader,
                dist_metric=dist_metric,
                normalize_feature=normalize_feature,
                visrank=visrank,
                visrank_topk=visrank_topk,
                save_dir=save_dir,
                use_metric_cuhk03=use_metric_cuhk03,
                ranks=ranks
            )
        return rank1
    def _k_reciprocal_neigh(self, initial_rank, i, k1):
        forward_k_neigh_index = initial_rank[i,:k1+1] #第i个图片的前20个相似图片的索引号
        backward_k_neigh_index = initial_rank[forward_k_neigh_index,:k1+1]
        fi = torch.where(backward_k_neigh_index==i)[0] #返回backward_k_neigh_index中等于i的图片的行索引号
        return forward_k_neigh_index[fi]  #返回与第i张图片 互相为k_reciprocal_neigh的图片索引号

    def _batch_torch_topk(self, qf, gf, k1, N=300):
        m = qf.shape[0]
        n = gf.shape[0]
        initial_rank = []
        for j in range(n // N + 1):
            temp_gf = gf[j * N:j * N + N]
            temp_qd = []
            for i in range(m // N + 1):
                temp_qf = qf[i * N:i * N + N]
                temp_d = 1.- temp_qf @ temp_gf.t()
                temp_qd.append(temp_d)
            temp_qd = torch.cat(temp_qd, dim=0)
            temp_qd = temp_qd / (torch.max(temp_qd, dim=0)[0])
            temp_qd = temp_qd.t()
            initial_rank.append(torch.topk(temp_qd, k=k1, dim=1, largest=False, sorted=True)[1])

            del temp_qd
            del temp_gf
            del temp_qf
            del temp_d
            gc.collect()
        torch.cuda.empty_cache()  # empty GPU memory
        initial_rank = torch.cat(initial_rank, dim=0)
        return initial_rank

    def _compute_rank(self, dist, k=15):
        _, index = dist.cpu().sort(-1,descending=False)
        return index[:,:k].cuda()

    def _constract_graph(self, feat, kk):
        initial_rank = self._batch_torch_topk(feat,feat,kk)
        #dist = cosine_distance(feat, feat)
        #initial_rank = self._compute_rank(dist)  
        edge = []
        print('******')
        for i in range(feat.size(0)):
            col = self._k_reciprocal_neigh(initial_rank, i, kk).unsqueeze(0)
            row = torch.ones_like(col)*i
            edge.append(torch.cat([col,row], dim=0))
        edge = torch.cat(edge, dim=1
        
        )
        return edge

    @torch.no_grad()
    def _evaluate_sss(self, epoch, dataset_name='', queryloader=None, galleryloader=None,
                  dist_metric='euclidean', normalize_feature=False, visrank=False,
                  visrank_topk=10, save_dir='', use_metric_cuhk03=False,ranks=[1,5,10,20]):
        print('Extracting features from CNN ... ')
        qf, q_pids, q_camids = self._feature_extraction(queryloader)
        q_num = qf.size(0)
        gf, g_pids, g_camids = self._feature_extraction(galleryloader)
        g_num = gf.size(0)
        

        precious = []
        K=20
        R=35
        self.model.eval()
        
        for k in range(1,K+1):
            feat = torch.cat((qf,gf),0).cuda()
            feat = F.normalize(feat, p=2, dim=1)
            edge_index = self._constract_graph(feat, k)
            for r in range(1,R+1):
                feat = self.model.getGarph(feat, edge_index)
                feat_n = F.normalize(feat, p=2, dim=1)
                q_feat_n = feat_n[:q_num]
                g_feat_n = feat_n[-g_num:]
                dist = q_feat_n @ g_feat_n.t()
                dist = 2.*torch.ones_like(dist)-dist
                distmat = dist.cpu().numpy()
                ranks = [1,5,10,20]
                print('k = {0}, r = {1}'.format(k, r))
                print('Computing CMC and mAP ...')
                cmc, mAP = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids)
                print('** Results **')
                precious.append(mAP)
                print('mAP: {:.1%}'.format(mAP))
                print('CMC curve')
                for r in ranks:
                    print('Rank-{:<3}:{:.1%}'.format(r,cmc[r-1]))
                    precious.append(cmc[r-1])
                mkdir_if_missing('market_02_edge/'.format(k))
                np.save('market_02_edge/k{0}.npy'.format(k), edge_index.cpu().numpy())
        precious = np.array(precious)
        precious = precious.reshape(K,R,5)
        np.save('market_alpha_02.npy',precious)
        return cmc[0]

    @torch.no_grad()
    def _evaluate(self, epoch, dataset_name='', queryloader=None, galleryloader=None,
                  dist_metric='euclidean', normalize_feature=False, visrank=False,
                  visrank_topk=10, save_dir='', use_metric_cuhk03=False,ranks=[1,5,10,20]):
        print('Extracting features from CNN ... ')
        qf, q_pids, q_camids = self._feature_extraction(queryloader)
        q_num = qf.size(0)
        gf, g_pids, g_camids = self._feature_extraction(galleryloader)
        g_num = gf.size(0)
        '''
        qfn = qf.cpu().numpy()
        qpidsfn = q_pids
        gfn = gf.cpu().numpy()
        gpidsfn = g_pids
        np.savez('feat_msmt.npz',qfn=qfn,qpidsfn=qpidsfn,gfn=gfn,gpidsfn=gpidsfn, qcamid=q_camids, gcamid =g_camids)
        '''
        f = torch.cat((qf,gf),0)
        print('Extracting features from GNN ... ')
        #affinity = euclidean_distance(f, f)
        #edge_index, _= init_edge(affinity, 15)
        edge_index = self._constract_graph(f, 15)
        print('Get edge')
        
        #affinity = euclidean_distance(f, f)
        #edge_index, _= init_edge(affinity, 15)
        
        edge_index = edge_index.cpu()
        self.model.eval()
        self.model=self.model.cpu()
        f = f.cpu()
        f = self.model.getGarph(f,edge_index)
        f = f.cuda()
        self.model = self.model.cuda()
        qf = f[:q_num]
        gf = f[-g_num:]
        
        qfn = qf.cpu().numpy()
        qpidsfn = q_pids
        gfn = gf.cpu().numpy()
        gpidsfn = g_pids
        #np.savez('feat_msmt_new.npz',qfn=qfn,qpidsfn=qpidsfn,gfn=gfn,gpidsfn=gpidsfn, qcamid=q_camids, gcamid =g_camids)
        
        
        print('Done, obtained {}-by-{} matrix'.format(qf.size(0), qf.size(1)))

        print('Extracting features from gallery set ...')
        print('Done, obtained {}-by-{} matrix'.format(gf.size(0), gf.size(1)))


        if normalize_feature:
            print('Normalzing features with L2 norm ...')
            #qf = F.normalize(qf, p=2, dim=1)
            #gf = F.normalize(gf, p=2, dim=1)

        print('Computing distance matrix with metric={} ...'.format(dist_metric))

        distmat = cosine_distance(qf, gf).cpu()
        distmat = distmat.numpy()
        print('Computing CMC and mAP ...')
        cmc, mAP = evaluate_rank(distmat, q_pids, g_pids, q_camids, g_camids, use_metric_cuhk03= use_metric_cuhk03)

        print('** Results **')
        print('mAP: {:.1%}'.format(mAP))
        print('CMC curve')
        for r in ranks:
            print('Rank-{:<3}:{:.1%}'.format(r,cmc[r-1]))
        
        if visrank:
            
            visualize_ranked_results(
                distmat,
                self.datamanger.return_testdataset_by_name(dataset_name),
                self.datamanger.data_type,
                width=self.datamanger.width,
                height=self.datamanger.height,
                save_dir=osp.join(save_dir, 'visrank_'+dataset_name),
                topk=visrank_topk
            )
          

        return cmc[0]
    
    @torch.no_grad()
    def visactmap(self, save_dir, width, height, print_freq,testloader=None):
        """Visualizes CNN activation maps to see where the CNN focuses on to extract features.

        This function takes as input the query images of target datasets

        Reference:
            - Zagoruyko and Komodakis. Paying more attention to attention: Improving the
              performance of convolutional neural networks via attention transfer. ICLR, 2017
            - Zhou et al. Omni-Scale Feature Learning for Person Re-Identification. ICCV, 2019.
        """
        if testloader == None:
            testloader = self.testloader
        self.model.eval()
        
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]
        GRID_SPACING = 10
        for target in list(testloader.keys()):
            queryloader = testloader[target]['query']
            # original images and activation maps are saved individually
            actmap_dir = osp.join(save_dir, 'actmap_'+target)
            mkdir_if_missing(actmap_dir)
            print('Visualizing activation maps for {} ...'.format(target))

            for batch_idx, data in enumerate(queryloader):
                imgs, paths = data[0], data[3]
                if self.use_gpu:
                    imgs = imgs.cuda()
                
                # forward to get convolutional feature maps
                try:
                    outputs = self.model(imgs, return_featuremaps=True)
                except TypeError:
                    raise TypeError('forward() got unexpected keyword argument "return_featuremaps". ' \
                                    'Please add return_featuremaps as an input argument to forward(). When ' \
                                    'return_featuremaps=True, return feature maps only.')
                
                if outputs.dim() != 4:
                    raise ValueError('The model output is supposed to have ' \
                                     'shape of (b, c, h, w), i.e. 4 dimensions, but got {} dimensions. '
                                     'Please make sure you set the model output at eval mode '
                                     'to be the last convolutional feature maps'.format(outputs.dim()))
                
                # compute activation maps
                outputs = (outputs**2).sum(1)
                b, h, w = outputs.size()
                outputs = outputs.view(b, h*w)
                outputs = F.normalize(outputs, p=2, dim=1)
                outputs = outputs.view(b, h, w)
                
                if self.use_gpu:
                    imgs, outputs = imgs.cpu(), outputs.cpu()

                for j in range(outputs.size(0)):
                    # get image name
                    path = paths[j]
                    imname = osp.basename(osp.splitext(path)[0])
                    
                    # RGB image
                    img = imgs[j, ...]
                    for t, m, s in zip(img, imagenet_mean, imagenet_std):
                        t.mul_(s).add_(m).clamp_(0, 1)
                    img_np = np.uint8(np.floor(img.numpy() * 255))
                    img_np = img_np.transpose((1, 2, 0)) # (c, h, w) -> (h, w, c)
                    
                    # activation map
                    am = outputs[j, ...].numpy()
                    am = cv2.resize(am, (width, height))
                    am = 255 * (am - np.max(am)) / (np.max(am) - np.min(am) + 1e-12)
                    am = np.uint8(np.floor(am))
                    am = cv2.applyColorMap(am, cv2.COLORMAP_JET)
                    
                    # overlapped
                    overlapped = img_np * 0.3 + am * 0.7
                    overlapped[overlapped>255] = 255
                    overlapped = overlapped.astype(np.uint8)

                    # save images in a single figure (add white spacing between images)
                    # from left to right: original image, activation map, overlapped image
                    grid_img = 255 * np.ones((height, 3*width+2*GRID_SPACING, 3), dtype=np.uint8)
                    grid_img[:, :width, :] = img_np[:, :, ::-1]
                    grid_img[:, width+GRID_SPACING: 2*width+GRID_SPACING, :] = am
                    grid_img[:, 2*width+2*GRID_SPACING:, :] = overlapped
                    cv2.imwrite(osp.join(actmap_dir, imname+'.jpg'), grid_img)

                if (batch_idx+1) % print_freq == 0:
                    print('- done batch {}/{}'.format(batch_idx+1, len(queryloader)))

    def _parse_data(self, data):
        imgs = data[0]
        labels = data[1]
        return imgs, labels

    def _extract_features(self, input):
        self.model.eval()
        return self.model(input)

    def _feature_extraction(self, data_loader):
        f_, pids_, camids_ = [], [], []
        for batch_idx, data in enumerate(data_loader):
            imgs, pids, camids = self._parse_data_for_eval(data)
            if self.use_gpu:
                imgs = imgs.cuda()
            features = self._extract_features(imgs)
            features = features.data.cpu()
            f_.append(features)
            pids_.extend(pids)
            camids_.extend(camids)
        f_ = torch.cat(f_, 0)
        pids_ = np.asarray(pids_)
        camids_ = np.asarray(camids_)
        return f_, pids_, camids_

    def _parse_data_for_eval(self, data):
        imgs = data[0]
        pids = data[1]
        camids = data[2]
        return imgs, pids, camids

    def _save_checkpoint(self, epoch, accuracy, save_dir, is_best=False):
        save_checkpoint({
            'state_dict':self.model.state_dict(),
            'epoch':epoch+1,
            'accuracy':accuracy,
            'optimizer':self.optimizer.state_dict(),
        }, save_dir, is_best=is_best)
