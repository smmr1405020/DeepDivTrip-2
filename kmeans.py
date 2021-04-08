import random
import math


def eucldist(p0, p1):
    dist = 0.0
    for i in range(0, len(p0)):
        dist += (p0[i] - p1[i]) ** 2
    return math.sqrt(dist)


# K-Means Algorithm
def kmeans(K, datapoints):
    # d - Dimensionality of Datapoints
    d = len(datapoints[0])

    # Limit our iterations
    Max_Iterations = 50

    cluster = [0] * len(datapoints)
    prev_cluster = [-1] * len(datapoints)

    # Randomly Choose Centers for the Clusters
    cluster_centers = []
    for j in range(0, K):
        cluster_centers += [random.choice(datapoints)]

    force_recalculation = True

    i = 0
    while (cluster != prev_cluster) or (i > Max_Iterations) or force_recalculation:

        prev_cluster = list(cluster)
        force_recalculation = False
        i += 1

        # Update Point's Cluster Allegiance
        for p in range(0, len(datapoints)):
            min_dist = math.inf

            for c in range(0, len(cluster_centers)):

                dist = eucldist(datapoints[p], cluster_centers[c])

                if dist < min_dist:
                    min_dist = dist
                    cluster[p] = c  # Reassign Point to new Cluster

        # Update Cluster's Position
        for c_cen in range(0, len(cluster_centers)):
            new_center = [0] * d
            members = 0
            for p in range(0, len(datapoints)):
                if cluster[p] == c_cen:  # If this point belongs to the cluster
                    for j in range(0, d):
                        new_center[j] += datapoints[p][j]
                    members += 1

            for j in range(0, d):
                if members != 0:
                    new_center[j] = new_center[j] / float(members)

                    # This means that our initial random assignment was poorly chosen
                # Change it to a new datapoint to actually force k clusters
                else:
                    new_center = random.choice(datapoints)
                    force_recalculation = True

            cluster_centers[c_cen] = new_center

    "======== Results ========"
    print("Clusters", cluster_centers)
    print("Iterations", i)
    print("Assignments", cluster)


# TESTING THE PROGRAM#
if __name__ == "__main__":
    # 2D - Datapoints List of n d-dimensional vectors. (For this example I already set up 2D Tuples)
    # Feel free to change to whatever size tuples you want...
    datapoints = [(3, 2), (2, 2), (1, 2), (0, 1), (1, 0), (1, 1), (5, 6), (7, 7), (9, 10), (11, 13), (12, 12), (12, 13),
                  (13, 13)]

    k = 2  # K - Number of Clusters

    kmeans(k, datapoints)
