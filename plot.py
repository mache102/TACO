import os
import json
import matplotlib.pyplot as plt

def plot_results(dataset, category="taco"):
    # Lambda values (folder names)
    lambdas = ["0.0004", "0.0008", "0.0016", "0.004", "0.009", "0.015"]
    bpp_list, lpips_list, psnr_list, msssim_list = [], [], [], []

    for lam in lambdas:
        folder = os.path.join("out", dataset, f"lambda_{lam}")
        stat_path = os.path.join(folder, "mean_stat.json")
        if not os.path.exists(stat_path):
            print(f"Warning: {stat_path} not found, skipping.")
            continue
        with open(stat_path, "r") as f:
            stats = json.load(f)
        bpp_list.append(stats["bpp"])
        lpips_list.append(stats["lpips"])
        psnr_list.append(stats["psnr"])
        msssim_list.append(stats["ms_ssim"])

    # Plot bpp vs lpips
    plt.figure()
    plt.plot(bpp_list, lpips_list, marker='o', label=category)
    plt.xlabel("BPP")
    plt.ylabel("LPIPS")
    plt.title("BPP vs LPIPS")
    plt.legend()
    plt.grid(True)
    plt.savefig("bpp_vs_lpips.png")

    # Plot bpp vs psnr
    plt.figure()
    plt.plot(bpp_list, psnr_list, marker='o', label=category)
    plt.xlabel("BPP")
    plt.ylabel("PSNR")
    plt.title("BPP vs PSNR")
    plt.legend()
    plt.grid(True)
    plt.savefig("bpp_vs_psnr.png")

    # Plot bpp vs ms-ssim
    plt.figure()
    plt.plot(bpp_list, msssim_list, marker='o', label=category)
    plt.xlabel("BPP")
    plt.ylabel("MS-SSIM")
    plt.title("BPP vs MS-SSIM")
    plt.legend()
    plt.grid(True)
    plt.savefig("bpp_vs_msssim.png")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--dataset", type=str, required=True, help="dataset (kodak or mscoco_val30k)")
    args = parser.parse_args()
    plot_results(args.dataset)