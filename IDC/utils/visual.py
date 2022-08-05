from matplotlib import pyplot as plt


class Plotter():
    def __init__(self, path, nepoch, idx=0):
        self.path = path
        self.data = {'epoch': [], 'acc_tr': [], 'acc_val': [], 'loss_tr': [], 'loss_val': []}
        self.nepoch = nepoch
        self.plot_freq = 10
        self.idx = idx

    def update(self, epoch, acc_tr, acc_val, loss_tr, loss_val):
        self.data['epoch'].append(epoch)
        self.data['acc_tr'].append(acc_tr)
        self.data['acc_val'].append(acc_val)
        self.data['loss_tr'].append(loss_tr)
        self.data['loss_val'].append(loss_val)

        if len(self.data['epoch']) % self.plot_freq == 0:
            self.plot()

    def plot(self, color='black'):
        fig, axes = plt.subplots(1, 4, figsize=(4 * 4, 3))
        fig.tight_layout(h_pad=3, w_pad=3)

        fig.suptitle(f"{self.path}", size=16, y=1.1)

        axes[0].plot(self.data['epoch'], self.data['acc_tr'], color, lw=0.8)
        axes[0].set_xlim([0, self.nepoch])
        axes[0].set_ylim([0, 100])
        axes[0].set_title('acc train')

        axes[1].plot(self.data['epoch'], self.data['acc_val'], color, lw=0.8)
        axes[1].set_xlim([0, self.nepoch])
        axes[1].set_ylim([0, 100])
        axes[1].set_title('acc val')

        axes[2].plot(self.data['epoch'], self.data['loss_tr'], color, lw=0.8)
        axes[2].set_xlim([0, self.nepoch])
        axes[2].set_ylim([0, 3])
        axes[2].set_title('loss train')

        axes[3].plot(self.data['epoch'], self.data['loss_val'], color, lw=0.8)
        axes[3].set_xlim([0, self.nepoch])
        axes[3].set_ylim([0, 3])
        axes[3].set_title('loss val')

        for ax in axes:
            ax.set_xlabel('epochs')

        plt.savefig(f'{self.path}/curve_{self.idx}.png', bbox_inches='tight')
        plt.close()