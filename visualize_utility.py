import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def generate_data(n):
    P1 = np.random.uniform(0,3,n)
    P2 = np.random.uniform(0,3,n)
    P3 = np.random.uniform(0,3,n)
    P4 = np.random.uniform(0,3,n)
    P5 = np.random.uniform(0,3,n)
    P6 = np.random.uniform(0,3,n)
    P7 = np.random.uniform(0,3,n)
    K1 =    0.5*np.exp(-(P1+1)**2/(2*P2**2))
    K2 =     np.exp(-(P1-1)**2/(2*P3**2))
    K3 =     np.exp(-(P4+K1)**2/(2*P5**2))
    K4 =     np.exp(-(P7+K2)**2/(2*P6**2))
    eps = 1e-5
    K1 += np.random.normal(0,eps,n); K2 += np.random.normal(0,eps,n)
    K3 += np.random.normal(0,eps,n); K4 += np.random.normal(0,eps,n)
    return np.column_stack([P1,P2,P3,P4,P5,P6,P7,K1,K2,K3,K4])

def main():
    np.random.seed(42)
    data = generate_data(1000)
    P1,P2,P3,P4,P5,P6,P7 = [data[:,i] for i in range(7)]
    K1,K2,K3,K4 = [data[:,i] for i in range(7,11)]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle('Generated Data Relationships', fontsize=16, fontweight='bold')
    cmap, alpha, s = 'viridis', 0.6, 10

    # K1 vs P1, P2
    ax1 = fig.add_subplot(2, 2, 1, projection='3d')
    sc1 = ax1.scatter(P1, P2, K1, c=K1, cmap=cmap, alpha=alpha, s=s)
    ax1.set_xlabel('P1'); ax1.set_ylabel('P2'); ax1.set_zlabel('K1')
    ax1.set_title('K1 vs P1, P2\n0.5·exp(-(P1+1)²/2P2²)')
    fig.colorbar(sc1, ax=ax1, shrink=0.5, label='K1')

    # K2 vs P1, P3
    ax2 = fig.add_subplot(2, 2, 2, projection='3d')
    sc2 = ax2.scatter(P1, P3, K2, c=K2, cmap=cmap, alpha=alpha, s=s)
    ax2.set_xlabel('P1'); ax2.set_ylabel('P3'); ax2.set_zlabel('K2')
    ax2.set_title('K2 vs P1, P3\nexp(-(P1-1)²/2P3²)')
    fig.colorbar(sc2, ax=ax2, shrink=0.5, label='K2')

    # K3 vs P4, P5, K1 — color encodes K1's influence
    ax3 = fig.add_subplot(2, 2, 3, projection='3d')
    sc3 = ax3.scatter(P4, P5, K3, c=K1, cmap=cmap, alpha=alpha, s=s)
    ax3.set_xlabel('P4'); ax3.set_ylabel('P5'); ax3.set_zlabel('K3')
    ax3.set_title('K3 vs P4, P5 (color = K1)\nexp(-(P4+K1)²/2P5²)')
    fig.colorbar(sc3, ax=ax3, shrink=0.5, label='K1')

    # K4 vs P6, P7, K2 — color encodes K2's influence
    ax4 = fig.add_subplot(2, 2, 4, projection='3d')
    sc4 = ax4.scatter(P7, P6, K4, c=K2, cmap=cmap, alpha=alpha, s=s)
    ax4.set_xlabel('P7'); ax4.set_ylabel('P6'); ax4.set_zlabel('K4')
    ax4.set_title('K4 vs P6, P7 (color = K2)\nexp(-(P7+K2)²/2P6²)')
    fig.colorbar(sc4, ax=ax4, shrink=0.5, label='K2')

    plt.tight_layout()
    plt.savefig('data_plots.png', dpi=150, bbox_inches='tight')
    plt.show()

main()