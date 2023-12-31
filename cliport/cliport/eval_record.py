"""Ravens main training script."""

import os
import pickle
import json

import numpy as np
import hydra
import pandas as pd
import torch
from PIL import Image

from cliport import agents
from cliport import dataset
from cliport import tasks
from cliport.utils import utils
from cliport.environments.environment import Environment


# Matplotlib for affordance
import matplotlib.pyplot as plt

@hydra.main(config_path='./cfg', config_name='eval')
def main(vcfg):
    # Load train cfg
    tcfg = utils.load_hydra_config(vcfg['train_config'])

    # Initialize environment and task.
    env = Environment(
        vcfg['assets_root'],
        disp=vcfg['disp'],
        shared_memory=vcfg['shared_memory'],
        hz=480,
        record_cfg=vcfg['record']
    )

    # Choose eval mode and task.
    mode = vcfg['mode']
    eval_task = vcfg['eval_task']
    if mode not in {'train', 'val', 'test'}:
        raise Exception("Invalid mode. Valid options: train, val, test")

    # Load eval dataset.
    dataset_type = vcfg['type']
    if 'multi' in dataset_type:
        ds = dataset.RavensMultiTaskDataset(vcfg['data_dir'],
                                            tcfg,
                                            group=eval_task,
                                            mode=mode,
                                            n_demos=vcfg['n_demos'],
                                            augment=False)
    else:
        ds = dataset.RavensDataset(os.path.join(vcfg['data_dir'], f"{eval_task}-{mode}"),
                                   tcfg,
                                   n_demos=vcfg['n_demos'],
                                   augment=False)

    all_results = {}
    name = '{}-{}-n{}'.format(eval_task, vcfg['agent'], vcfg['n_demos'])

    # Save path for results.
    json_name = f"multi-results-{mode}.json" if 'multi' in vcfg['model_path'] else f"results-{mode}.json"
    save_path = vcfg['save_path']
    print(f"Save path for results: {save_path}")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    save_json = os.path.join(save_path, f'{name}-{json_name}')

    # Load existing results.
    existing_results = {}
    if os.path.exists(save_json):
        with open(save_json, 'r') as f:
            existing_results = json.load(f)

    # Make a list of checkpoints to eval.
    ckpts_to_eval = list_ckpts_to_eval(vcfg, existing_results)

    # added by Yusuke 24/08/23, this is where /data is located. /data is generated by cliport/demo.py
    #print(vcfg['data_dir'])
    #data_path = os.path.join(vcfg['data_dir'], "{}-{}".format(eval_task, mode))
    data_path = os.path.join('data', "{}-{}".format(eval_task, mode)) # we assume the root_dir where you run this file is located {root_dir}/data

    # Evaluation loop
    print(f"Evaluating: {str(ckpts_to_eval)}")
    for ckpt in ckpts_to_eval:
        model_file = os.path.join(vcfg['model_path'], ckpt)

        if not os.path.exists(model_file) or not os.path.isfile(model_file):
            print(f"Checkpoint not found: {model_file}")
            continue
        elif not vcfg['update_results'] and ckpt in existing_results:
            print(f"Skipping because of existing results for {model_file}.")
            continue

        results = []
        mean_reward = 0.0

        # Run testing for each training run.
        for train_run in range(vcfg['n_repeats']):

            # Initialize agent.
            utils.set_seed(train_run, torch=True)
            agent = agents.names[vcfg['agent']](name, tcfg, None, ds)

            # Load checkpoint
            agent.load(model_file)
            print(f"Loaded: {model_file}")

            record = vcfg['record']['save_video']
            n_demos = vcfg['n_demos']

            # metadata will be saved into csv later
            metadata = []
            total_done = 0

            # Run testing and save total rewards with last transition info.
            for i in range(0, n_demos):
                print(f'Test: {i + 1}/{n_demos}')
                episode, seed = ds.load(i)
                goal = episode[-1]
                total_reward = 0
                np.random.seed(seed)

                # set task
                if 'multi' in dataset_type:
                    task_name = ds.get_curr_task()
                    task = tasks.names[task_name]()
                    print(f'Evaluating on {task_name}')
                else:
                    task_name = vcfg['eval_task']
                    task = tasks.names[task_name]()

                task.mode = mode
                env.seed(seed)
                env.set_task(task)
                obs = env.reset(color_change = vcfg['random_BackGroundColor'])
                info = env.info
                reward = 0

                # keep the copy of img to record if the task fails
                obs_copy = obs.copy()


                # Start recording video (NOTE: super slow)
                if record:
                    video_name = f'{task_name}-{i+1:06d}'
                    if 'multi' in vcfg['model_task']:
                        video_name = f"{vcfg['model_task']}-{video_name}"
                    env.start_rec(video_name)

                for _ in range(task.max_steps):
                    act = agent.act(obs, info, goal)
                    lang_goal = info['lang_goal']
                    print(f'Lang Goal: {lang_goal}')
                    obs, reward, done, info = env.step(act)
                    total_reward += reward
                    print(f'Total Reward: {total_reward:.3f} | Done: {done}\n')
                    if done:
                        total_done += done
                        break

                results.append((total_reward, info))
                mean_reward = np.mean([r for r, i in results])
                print(f'Mean: {mean_reward} | Total Successful Demos: {total_done} | Task: {task_name} | Ckpt: {ckpt}')
                
                #record the run, later this will be improved with generative AI
                # Extract the colormap from the image and save it
                img = torch.from_numpy(ds.process_sample((obs_copy, None, None, info), augment=False)['img'])                     
                colormap = np.array(img.detach().cpu().numpy())[:,:,:3]
                # Swap the width and height dimensions, Convert the colormap to the appropriate data type, Create a PIL image from the colormap array
                colormap_image = Image.fromarray(np.uint8(np.transpose(colormap, (1, 0, 2))))
                # Save the colormap image as a PNG file
                img_dir = os.path.join(data_path, 'raw_images') 
                # Create the directory if it does not exist
                if not os.path.exists(img_dir):
                    os.makedirs(img_dir)
                if eval_task == 'packing-unseen-google-object' or eval_task == 'packing-seen-google-object':
                    # Replace spaces with underscores in the object name
                    obj_name = task.object_name
                    obj_name_tosave = obj_name.replace(' ', '_')
                elif eval_task == 'put-block-in-bowl-test-colors' or 'put-block-in-bowl-seen-colors':
                    # extract color
                    pick_color = lang_goal.split()[2]
                    place_color = lang_goal.split()[6]
                    obj_name = None
                    obj_name_tosave = pick_color + '_' + place_color
                elif eval_task == 'separating-piles-full':
                    block_color = lang_goal.split()[4]
                    square_color = lang_goal.split()[8]
                    obj_name = None
                    obj_name_tosave = block_color + '_' + square_color
                elif eval_task == 'packing-shapes':
                    obj_name = lang_goal.split()[2]
                    obj_name_tosave = obj_name

                    
                colormap_path = os.path.join(img_dir, f'{seed}_{obj_name_tosave}.png')
                colormap_image.save(colormap_path)
                
                # prepare empty directory for image editing later, apply this only if the task failed
                if not done: 
                    edited_imgs_dir = os.path.join(data_path, 'edited_images')
                    # Create the directory if it does not exist
                    if not os.path.exists(edited_imgs_dir):
                        os.makedirs(edited_imgs_dir)
                    # also create directory for each images 
                    edited_img_dir = os.path.join(edited_imgs_dir, f'{seed}_{obj_name_tosave}')     
                    if not os.path.exists(edited_img_dir):
                        os.makedirs(edited_img_dir)

                if vcfg['save_affordance']:
                    
                    # get pixel location of pick and place
                    pick, place = act['pick'], act['place']  

                    # Create dir and run affordances
                    img_affordance = ds.process_sample((obs_copy, None, None, info), augment=False)['img']
                    affordance_imgs_dir = os.path.join(data_path, 'affordances')

                    if not os.path.exists(affordance_imgs_dir):
                        os.makedirs(affordance_imgs_dir)

                    affordance_fail_dir = os.path.join(data_path, 'affordances/fail')
                    if not os.path.exists(affordance_fail_dir):
                        os.makedirs(affordance_fail_dir)
                    
                    affordance_success_dir = os.path.join(data_path, 'affordances/success')
                    if not os.path.exists(affordance_success_dir):
                        os.makedirs(affordance_success_dir)

                    # Save affordances
                    if not done:
                        affordance_path = os.path.join(affordance_imgs_dir, f'fail/{seed}_{obj_name_tosave}.png')
                    else:
                        affordance_path = os.path.join(affordance_imgs_dir, f'success/{seed}_{obj_name_tosave}.png')
                    
                    # affordance_path = os.path.join(affordance_imgs_dir, f'{seed}_{obj_name_tosave}.png')

                    run_affordance(img_affordance, lang_goal, agent, pick, place, affordance_path, draw_grasp_lines = True, affordance_heatmap_scale=30, alpha_lvl = vcfg['alpha_lvl'])
                
                # record above information for image editing, text mapping later in csv
                demo_data = {
                                'lang_goal': lang_goal,
                                'object_name': obj_name,
                                'img_path': colormap_path,
                                'seed': seed,
                                'success': done
                            }
                metadata.append(demo_data)

                # End recording video
                if record:
                    env.end_rec()

            # write metadata into CSV
            csv_file_path = os.path.join(data_path, 'metadata.csv')
            df = pd.DataFrame.from_dict(metadata)
            df.to_csv(csv_file_path, index=False)

                

            all_results[ckpt] = {
                'episodes': results,
                'mean_reward': mean_reward,
            }

        # Save results in a json file.
        if vcfg['save_results']:

            # Load existing results
            if os.path.exists(save_json):
                with open(save_json, 'r') as f:
                    existing_results = json.load(f)
                existing_results.update(all_results)
                all_results = existing_results

            with open(save_json, 'w') as f:
                json.dump(all_results, f, indent=4)


def list_ckpts_to_eval(vcfg, existing_results):
    ckpts_to_eval = []

    # Just the last.ckpt
    if vcfg['checkpoint_type'] == 'last':
        last_ckpt = 'last.ckpt'
        ckpts_to_eval.append(last_ckpt)

    # Validation checkpoints that haven't been already evaluated.
    elif vcfg['checkpoint_type'] == 'val_missing':
        checkpoints = sorted([c for c in os.listdir(vcfg['model_path']) if "steps=" in c])
        ckpts_to_eval = [c for c in checkpoints if c not in existing_results]

    # Find the best checkpoint from validation and run eval on the test set.
    elif vcfg['checkpoint_type'] == 'test_best':
        result_jsons = [c for c in os.listdir(vcfg['results_path']) if "results-val" in c]
        if 'multi' in vcfg['model_task']:
            result_jsons = [r for r in result_jsons if "multi" in r]
        else:
            result_jsons = [r for r in result_jsons if "multi" not in r]

        if len(result_jsons) > 0:
            result_json = result_jsons[0]
            with open(os.path.join(vcfg['results_path'], result_json), 'r') as f:
                eval_res = json.load(f)
            best_checkpoint = 'last.ckpt'
            best_success = -1.0
            for ckpt, res in eval_res.items():
                if res['mean_reward'] > best_success:
                    best_checkpoint = ckpt
                    best_success = res['mean_reward']
            print(best_checkpoint)
            ckpt = best_checkpoint
            ckpts_to_eval.append(ckpt)
        else:
            print("No best val ckpt found. Using last.ckpt")
            ckpt = 'last.ckpt'
            ckpts_to_eval.append(ckpt)

    # Load a specific checkpoint with a substring e.g: 'steps=10000'
    else:
        print(f"Looking for: {vcfg['checkpoint_type']}")
        checkpoints = [c for c in os.listdir(vcfg['model_path']) if vcfg['checkpoint_type'] in c]
        checkpoint = checkpoints[0] if len(checkpoints) > 0 else ""
        ckpt = checkpoint
        ckpts_to_eval.append(ckpt)

    return ckpts_to_eval


def save_colormap_image(colormap, img_path):
    colormap_image = Image.fromarray(colormap)
    colormap_image.save(img_path)
        
def run_affordance(img, lang_goal, agent, pick, place, affordance_path, draw_grasp_lines=True, affordance_heatmap_scale=30, alpha_lvl=0.4):
        # Get color and depth inputs
        img = torch.from_numpy(img)
        color = np.uint8(img.detach().cpu().numpy())[:,:,:3]
        color = color.transpose(1,0,2)
        depth = np.array(img.detach().cpu().numpy())[:,:,3]
        depth = depth.transpose(1,0)
        
        # Visualize pick affordance
        pick_inp = {'inp_img': img, 'lang_goal': str(lang_goal)}
        pick_conf = agent.attn_forward(pick_inp)
        logits = pick_conf.detach().cpu().numpy()

        pick_conf = pick_conf.detach().cpu().numpy()
        argmax = np.argmax(pick_conf)
        argmax = np.unravel_index(argmax, shape=pick_conf.shape)
        p0 = argmax[:2]
        p0_theta = (argmax[2] * (2 * np.pi / pick_conf.shape[2])) * -1.0
    
        line_len = 30
        pick0 = (pick[0] + line_len/2.0 * np.sin(p0_theta), pick[1] + line_len/2.0 * np.cos(p0_theta))
        pick1 = (pick[0] - line_len/2.0 * np.sin(p0_theta), pick[1] - line_len/2.0 * np.cos(p0_theta))
        
        # Visualize place affordance
        place_inp = {'inp_img': img, 'p0': pick, 'lang_goal': str(lang_goal)}
        place_conf = agent.trans_forward(place_inp)

        place_conf = place_conf.permute(1, 2, 0)
        place_conf = place_conf.detach().cpu().numpy()
        argmax = np.argmax(place_conf)
        argmax = np.unravel_index(argmax, shape=place_conf.shape)
        p1_pix = argmax[:2]
        p1_theta = (argmax[2] * (2 * np.pi / place_conf.shape[2]) + p0_theta) * -1.0
        
        line_len = 30
        place0 = (place[0] + line_len/2.0 * np.sin(p1_theta), place[1] + line_len/2.0 * np.cos(p1_theta))
        place1 = (place[0] - line_len/2.0 * np.sin(p1_theta), place[1] - line_len/2.0 * np.cos(p1_theta))
        
        # Overlay affordances on RGB input
        pick_logits_disp = np.uint8(logits * 255 * affordance_heatmap_scale).transpose(1,0,2)
        place_logits_disp = np.uint8(np.sum(place_conf, axis=2)[:,:,None] * 255 * affordance_heatmap_scale).transpose(1,0,2)    
        
        pick_logits_disp_masked = np.ma.masked_where(pick_logits_disp < 0, pick_logits_disp)
        place_logits_disp_masked = np.ma.masked_where(place_logits_disp < 0, place_logits_disp)
        
        ### PLOTTING HERE AND SAVING
        fig = plt.figure(figsize=(13, 7))
        plt.imshow(color)
        plt.axis('off')

        if draw_grasp_lines:
            plt.plot((pick1[0], pick0[0]), (pick1[1], pick0[1]), color='r', linewidth=1)
            plt.plot((place1[0], place0[0]), (place1[1], place0[1]), color='g', linewidth=1)

        plt.imshow(pick_logits_disp_masked.squeeze(), alpha=alpha_lvl)
        plt.imshow(place_logits_disp_masked.squeeze(), cmap='viridis', alpha=alpha_lvl)
        fig.savefig(affordance_path)      


if __name__ == '__main__':
    main()
