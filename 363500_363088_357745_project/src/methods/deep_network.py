import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

## MS2


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes, layer1 = 512, layer2 = 280, layer3 = 120, layer4 = 80, device=torch.device('cpu')):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)
        
        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
        """
        super(MLP, self).__init__()

        self.lin1 = nn.Linear(input_size, layer1);
        self.lin2 = nn.Linear(layer1, layer2);
        self.lin3 = nn.Linear(layer2, layer3);
        self.lin4 = nn.Linear(layer3, layer4);
        self.lin5 = nn.Linear(layer4, n_classes);
        self.device = device

        
    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
            
        #preds = x.flatten(- x.ndim + 1)
        ##preds = x.reshape((x.shape[0], -1))
        #preds = x.view(x.size(0), -1)


        #model1
        # preds = F.relu(self.lin1(x))
        # preds = F.sigmoid(self.lin2(preds))
        # preds = F.relu(self.lin3(preds))
        #model2
        # preds = F.relu(self.lin1(x))
        # preds = F.relu(self.lin2(preds))
        # preds = F.relu(self.lin3(preds))
        #model3
        # preds = F.sigmoid(self.lin1(x))
        # preds = F.sigmoid(self.lin2(preds))
        # preds = F.sigmoid(self.lin3(preds))
        #model4
        # preds = F.sigmoid(self.lin1(x))
        # preds = F.sigmoid(self.lin2(preds))
        # preds = F.relu(self.lin3(preds))
        #model5
        # preds = F.sigmoid(self.lin1(x))
        # preds = F.relu(self.lin2(preds))
        # preds = F.relu(self.lin3(preds))
        #model6
        # preds = F.relu(self.lin1(x))
        # preds = F.sigmoid(self.lin2(preds))
        # preds = F.relu(self.lin3(preds))
        # preds = F.sigmoid(self.lin4(preds))
        #model7
        preds = F.relu(self.lin1(x))
        preds = F.relu(self.lin2(preds))
        preds = F.relu(self.lin3(preds))
        preds = F.relu(self.lin4(preds))
        #model8
        # preds = F.relu(self.lin1(x))
        # preds = F.sigmoid(self.lin2(preds))
        # preds = F.sigmoid(self.lin3(preds))
        # preds = F.relu(self.lin4(preds))




        preds = self.lin5(preds)

        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes, device=torch.device('cpu')):
        """
        Initialize the network.
        
        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)
        
        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super(CNN, self).__init__()
        self.conv2d1 = nn.Conv2d(input_channels, 3, 3, stride=1, padding=1)
        self.conv2d2 = nn.Conv2d(3,  12,  3, stride = 1, padding = 1)
        self.fc1 = nn.Linear(12*7*7, 120)
        self.fc2 = nn.Linear(120, n_classes)
        self.device = device

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = F.max_pool2d(F.relu(self.conv2d1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2d2(x)), 2)
        x = x.reshape((x.shape[0], -1))
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class MyViT(nn.Module):
    def get_positional_embeddings(sequence_length, d):
        positions = torch.arange(sequence_length, dtype=torch.float32).unsqueeze(1)
        indices = torch.arange(d, dtype=torch.float32)
        angle_rates = 1 / torch.pow(10000, (2 * (indices // 2)) / d)
        angle_rads = positions * angle_rates

        sines = torch.sin(angle_rads[:, 0::2])
        cosines = torch.cos(angle_rads[:, 1::2])

        pos_encoding = torch.zeros(sequence_length, d)
        pos_encoding[:, 0::2] = sines
        pos_encoding[:, 1::2] = cosines
        return pos_encoding
    def patchify(images, n_patches, device):
        n, c, h, w = images.shape
        assert h == w, "Height and Width must be equal"
        patch_size = h // n_patches
        patches = images.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        patches = patches.view(n, n_patches * n_patches, c * patch_size * patch_size)
        return patches.to(device)
    """
    A Transformer-based neural network
    """

    def __init__(self, chw, n_patches=7, n_blocks=8, hidden_d=8, n_heads=8, out_d=10, device=torch.device('cpu')):
        """
        Initialize the network.
        
        """
        super().__init__()
        self.chw = chw 
        self.n_patches = n_patches
        self.n_blocks = n_blocks
        self.n_heads = n_heads
        self.hidden_d = hidden_d


        assert chw[2] % n_patches == 0 
        assert chw[3] % n_patches == 0
        self.patch_size =  (chw[2]/n_patches, chw[3]/n_patches)

        self.input_d = int(chw[1] * self.patch_size[1] * self.patch_size[1])
        self.linear_mapper = nn.Linear(self.input_d, self.hidden_d)

        self.class_token = nn.Parameter(torch.rand(1, self.hidden_d))

        self.positional_embeddings = MyViT.get_positional_embeddings(n_patches ** 2 + 1, self.hidden_d).to(device=device)

        self.blocks = nn.ModuleList([MyViT.MyViTBlock(hidden_d, n_heads) for _ in range(n_blocks)])

        self.mlp = MLP(input_size=hidden_d, n_classes=out_d, layer1=128, layer2=32, layer3=16)
        self.device = device

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        n, _, _, _ = x.shape

        patches = MyViT.patchify(x,self.n_patches, self.device)

        tokens = self.linear_mapper(patches)

        tokens = torch.cat((self.class_token.expand(n, 1, -1), tokens), dim=1)

        pos_embedding = self.positional_embeddings.repeat(n, 1 ,1)
        out = tokens + pos_embedding

        for block in self.blocks:
            out = block(out)

        out = out[:, 0]

        out = self.mlp(out)

        return out
    

    ## Multi-head self attention
    class MyMSA(nn.Module):
        def __init__(self, d, n_heads):
            super(MyViT.MyMSA, self).__init__()
            self.d = d
            self.n_heads = n_heads

            assert d % n_heads == 0, f"Can't divide dimension {d} into {n_heads} heads"
            d_head = int(d / n_heads)
            self.d_head = d_head 
            # self.multihead = nn.MultiheadAttention(d, num_heads=n_heads, dropout=0.004)

            self.qkv = nn.Linear(d, 3 * d)
            self.softmax = nn.Softmax(dim=-1)
            self.fc_out = nn.Linear(d, d)

        def forward(self, sequences):
            batch_size, seq_length, _ = sequences.shape

            qkv = self.qkv(sequences) # (batch_size, seq_length, 3 * d)
            qkv = qkv.view(batch_size, seq_length, 3, self.n_heads, self.d_head)
            q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]  # each (batch_size, seq_length, n_heads, d_head)

            q = q.permute(0, 2, 1, 3)
            k = k.permute(0, 2, 1, 3)
            v = v.permute(0, 2, 1, 3)

            scores = torch.matmul(q, k.transpose(-2, -1)) / (self.d_head ** 0.5)

            attn = self.softmax(scores)
            # attn = self.multihead(q,k,v)

            out = torch.matmul(attn, v)
            out = out.permute(0, 2, 1, 3).contiguous()
            out = out.view(batch_size, seq_length, -1)

            out = self.fc_out(out)
            return out
    class MyViTBlock(nn.Module):
        def __init__(self, hidden_d, n_heads, mlp_ratio=4):
            super(MyViT.MyViTBlock, self).__init__()
            self.hidden_d = hidden_d
            self.n_heads = n_heads

            self.norm1 = nn.LayerNorm(hidden_d)
            self.mhsa = MyViT.MyMSA(hidden_d, n_heads) 
            self.norm2 = nn.LayerNorm(hidden_d)
            self.mlp = nn.Sequential( 
                nn.Linear(hidden_d, mlp_ratio * hidden_d),
                nn.GELU(),
                nn.Linear(mlp_ratio * hidden_d, hidden_d)
            )

        def forward(self, x):
            out = self.mhsa(self.norm1(x)) + x
            out = self.mlp(self.norm2(out)) + out
            return out
class Trainer(object):
    """
    Trainer class for the deep networks.

    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size, device=torch.device('cpu')):
        """
        Initialize the trainer object for a given model.

        Arguments:
            model (nn.Module): the model to train
            lr (float): learning rate for the optimizer
            epochs (int): number of epochs of training
            batch_size (int): number of data points in each batch
        """
        self.device = device
        self.lr = lr
        self.epochs = epochs
        self.model = model.to(self.device)
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss().to(self.device)
        self.optimizer = torch.optim.SGD(params= model.parameters(), lr= lr)  ### WRITE YOUR CODE HERE

    def train_all(self, dataloader):
        """
        Fully train the model over the epochs. 
        
        In each epoch, it calls the functions "train_one_epoch". If you want to
        add something else at each epoch, you can do it here.

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader,ep=ep)
            ### WRITE YOUR CODE HERE if you want to do add something else at each epoch

    def train_one_epoch(self, dataloader, ep):
        """
        Train the model for ONE epoch.

        Should loop over the batches in the dataloader. (Recall the exercise session!)
        Don't forget to set your model to training mode, i.e., self.model.train()!

        Arguments:
            dataloader (DataLoader): dataloader for training data
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        self.model.train()
        max_iter = len(dataloader)
        for it, (x, y) in enumerate(dataloader):
            x = x.to(self.device)
            y = y.to(self.device).long()
            # Ensure y is of type torch.int64

            logits = self.model.forward(x)

            loss = self.criterion(logits, y)

            loss.backward()

            self.optimizer.step()

            self.optimizer.zero_grad()
            print(f"\rEpoch [{ep+1}/{self.epochs}] Iteration [{it+1}/{max_iter}]", end='')

    def predict_torch(self, dataloader):
        """
        Predict the validation/test dataloader labels using the model.

        Hints:
            1. Don't forget to set your model to eval mode, i.e., self.model.eval()!
            2. You can use torch.no_grad() to turn off gradient computation, 
            which can save memory and speed up computation. Simply write:
                with torch.no_grad():
                    # Write your code here.

        Arguments:
            dataloader (DataLoader): dataloader for validation/test data
        Returns:
            pred_labels (torch.tensor): predicted labels of shape (N,),
                with N the number of data points in the validation/test data.
        """
        ##
        ###
        #### WRITE YOUR CODE HERE!
        ###
        ##
        self.model.eval()
        pred_labels = torch.tensor([], dtype=torch.long, device=self.device)
        with torch.no_grad():
            for batch in dataloader:
                x = batch[0].to(self.device) ## It is a list of ONE element
                logits = self.model.forward(x)
                pred_labels = torch.cat((pred_labels, torch.argmax(logits,dim=1)))
        return pred_labels
    
    def fit(self, training_data, training_labels):
        """
        Trains the model, returns predicted labels for training data.

        This serves as an interface between numpy and pytorch.

        Arguments:
            training_data (array): training data of shape (N,D)
            training_labels (array): regression target of shape (N,)
        Returns:
            pred_labels (array): target of shape (N,)
        """

        # First, prepare data for pytorch
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(), 
                                      torch.from_numpy(training_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        
        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        """
        Runs prediction on the test data.

        This serves as an interface between numpy and pytorch.
        
        Arguments:
            test_data (array): test data of shape (N,D)
        Returns:
            pred_labels (array): labels of shape (N,)
        """
        # First, prepare data for pytorch
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)

        # We return the labels after transforming them into numpy array.
        return pred_labels.cpu().numpy()