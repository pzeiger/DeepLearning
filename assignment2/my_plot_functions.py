import matplotlib.pyplot as plt

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']


def plot_acc_loss_ww(log, size_epoch):

    fig, axs = plt.subplots(2, figsize=(10,8), sharex=True)

    axins = [None, None]

    train_batch = log.where(log.dataset == 'train_batch').dropna(subset=('criterion_mean', 'loss'))
    train = log.where(log.dataset == 'train').dropna(subset=('criterion_mean', 'loss'))
    test = log.where(log.dataset == 'test').dropna(subset=('criterion_mean', 'loss'))

    axs[0].scatter(train_batch.iteration/size_epoch, train_batch.criterion_mean,
                   label='training data per batch', alpha=.1, color=colors[7])
    axs[0].plot(train.epoch, train.criterion_mean, '^-', label='all training data',
                alpha=1, lw=4, color=colors[0], markersize=8)
    axs[0].plot(test.epoch, test.criterion_mean, 'v--', label='all test data',
                alpha=1, lw=3, color=colors[1], markersize=8)
    axs[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=14)
    axs[0].set_ylim(bottom=0., top=1.)
    axs[0].set_ylabel('Sorensen Dice Score', fontsize=14)
    axs[0].tick_params(labelsize=12)


    axs[1].scatter(train_batch.iteration/size_epoch, train_batch.loss,
                   label='training data per batch', alpha=.1, color=colors[7])
    axs[1].plot(train.epoch, train.loss, '^-', label='all training data',
                alpha=1, lw=4, color=colors[0], markersize=8)
    axs[1].plot(test.epoch, test.loss, 'v--', label='all test data',
                alpha=.8, lw=3, color=colors[1], markersize=8)
    axs[1].set_ylabel('Loss $J$', fontsize=14)
    axs[1].tick_params(labelsize=12)
    axs[1].set_xlabel('Epochs', fontsize=14)
    axs[1].set_xlim(left=-0.3, right=test.epoch.max()+0.5)

    fig.tight_layout()
    return fig, axs, axins



def plot_acc_loss_MNIST(log, size_epoch, axins_on=True):

    fig, axs = plt.subplots(2, figsize=(10,8), sharex=True)

    axins = [None, None]

    train_batch = log.where(log.dataset == 'train_batch').dropna(subset=('criterion_mean', 'loss'))
    train = log.where(log.dataset == 'train').dropna(subset=('criterion_mean', 'loss'))
    test = log.where(log.dataset == 'test').dropna(subset=('criterion_mean', 'loss'))

    axs[0].scatter(train_batch.iteration/size_epoch, train_batch.criterion_mean*100,
                   label='training data per batch', alpha=.1, color=colors[7])
    axs[0].plot(train.epoch, train.criterion_mean*100, '^-', label='all training data',
                alpha=1, lw=4, color=colors[0], markersize=8)
    axs[0].plot(test.epoch, test.criterion_mean*100, 'v--', label='all test data',
                alpha=1, lw=3, color=colors[1], markersize=8)
    axs[0].legend(loc='lower center', bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=14)
    axs[0].set_ylim(bottom=0., top=100.)
    axs[0].set_ylabel('Accuracy in %', fontsize=14)
    axs[0].tick_params(labelsize=12)


    if axins_on:
        axins[0] = axs[0].inset_axes([0.2, 0.1, 0.7, 0.3])
        axins[0].scatter(train_batch.iteration/size_epoch, train_batch.criterion_mean*100,
                         label='training data per batch', alpha=.1, color=colors[7])
        axins[0].plot(train.epoch, train.criterion_mean*100, '^-', label='all training data',
                      alpha=1, lw=4, color=colors[0], markersize=8)
        axins[0].plot(test.epoch, test.criterion_mean*100, 'v--', label='all test data',
                      alpha=1, lw=3, color=colors[1], markersize=8)
        axins[0].set_xlim(0, test.epoch.max()+0.5)
        axins[0].set_ylim(95, 100)
        axs[0].indicate_inset_zoom(axins[0], edgecolor="black")


    axs[1].scatter(train_batch.iteration/size_epoch, train_batch.loss,
                   label='training data per batch', alpha=.1, color=colors[7])
    axs[1].plot(train.epoch, train.loss, '^-', label='all training data',
                alpha=1, lw=4, color=colors[0], markersize=8)
    axs[1].plot(test.epoch, test.loss, 'v--', label='all test data',
                alpha=.8, lw=3, color=colors[1], markersize=8)
    axs[1].set_ylabel('Loss', fontsize=14)
    axs[1].tick_params(labelsize=12)
    axs[1].set_xlabel('Epochs', fontsize=14)
    axs[1].set_xlim(left=-0.3, right=test.epoch.max()+0.5)

    if axins_on:
        axins[1] = axs[1].inset_axes([0.2, 0.6, 0.7, 0.3])
        axins[1].scatter(train_batch.iteration/size_epoch, train_batch.loss,
                         label='training data per batch', alpha=.1, color=colors[7])
        axins[1].plot(train.epoch, train.loss, '^-', label='all training data',
                      alpha=1, lw=4, color=colors[0], markersize=8)
        axins[1].plot(test.epoch, test.loss, 'v--', label='all test data',
                      alpha=1, lw=3, color=colors[1], markersize=8)
        axins[1].set_xlim(0, test.epoch.max()+0.5)
        axins[1].set_ylim(-0.1, 0.5)
        axs[1].indicate_inset_zoom(axins[1], edgecolor="black")

    fig.tight_layout()
    return fig, axs, axins



