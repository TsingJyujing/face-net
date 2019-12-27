from fnet.sample.wider import bbox_label_file_loader

if __name__ == '__main__':
    r = bbox_label_file_loader("data/wider/wider_face_split/wider_face_train_bbx_gt.txt")
    print("There're {} images.".format(len(r)))
    print("Include {} faces".format(sum(
        len(v) for k, v in r.items()
    )))
