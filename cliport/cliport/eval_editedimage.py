"""Ravens main training script."""

import os
import pickle
import json

from PIL import Image
import random
import pandas as pd
import torch
from datetime import datetime

import numpy as np
import hydra
from cliport import agents
from cliport import dataset
from cliport import tasks
from cliport.utils import utils
from cliport.environments.environment import Environment

from cliport.eval_record import run_affordance


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

    # Make new affordance folder 
    data_path = os.path.join(vcfg['data_dir'], f"{eval_task}-{mode}")
    start_datetime = datetime.now().strftime("%d.%m.%Y_%H.%M.%S")
    affordance_imgs_dir = os.path.join(data_path, "affordances", start_datetime)
    os.mkdir(affordance_imgs_dir)

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

        # Initialize task
        if 'multi' in dataset_type:
            task_name = ds.get_curr_task()
            task = tasks.names[task_name]()
        else:
            task_name = vcfg['eval_task']
            task = tasks.names[task_name]()

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

            # Load CSV file for lang goals
            csv_file_path = os.path.join(vcfg['data_dir'], f'{eval_task}-{mode}', 'metadata.csv')
            df = pd.read_csv(csv_file_path)

            # Filter the DataFrame based on the condition
            fail_rows = df[df['success'] == False]


            

            success_rate_demo_list = []

            # Run failed runs in eval_record.py with edited images
            for i, (index, row) in enumerate(fail_rows.iterrows()):
                if i + 1 > n_demos:
                    break
                print(f'Test: {i + 1}/{n_demos}')
                episode, seed = ds.load(i)
                goal = episode[-1]
                total_reward = 0
                total_done = 0
    
                original_lang_goal = row['lang_goal']
                object_name = row['object_name']
                image_path = row['img_path']
                seed = row['seed']

                print(f"Lang Goal for {image_path}: {original_lang_goal}")

                if eval_task == 'packing-seen-google-object':
                    # setup lang goal, do not change the object name
                    lang_goal = original_lang_goal

                    # define foldername
                    foldername = str(seed) + '_' + object_name.replace(' ','_')
                
                # modify the lang goal from unseen to seen
                elif eval_task == 'packing-unseen-google-object':
                    # dictionary to map unseen to seen object. This is obtained from prompt modification notebook
                    unseen2seen_map = {'ball puzzle': 'crayon box', 'black and blue sneakers': 'black shoe with orange stripes', 'black shoe with green stripes': 'black shoe with orange stripes', 'brown fedora': 'black fedora', 'dinosaur figure': 'rhino figure', 'hammer': 'tablet', 'light brown boot with golden laces': 'black boot with leopard print', 'lion figure': 'rhino figure', 'pepsi max box': 'pepsi gold caffeine free box', 'pepsi next box': 'pepsi wild cherry box', 'porcelain salad plate': 'porcelain cup', 'porcelain spoon': 'porcelain cup', 'red and white striped towel': 'green and white striped towel', 'red cup': 'porcelain cup', 'screwdriver': 'scissors', 'toy train': 'toy school bus', 'unicorn toy': 'android toy', 'white razer mouse': 'black razer mouse', 'yoshi figure': 'mario figure'}
                
                    # task goal format 
                    pick_google_object = 'pick {} into a brown box'

                    # setup new lang goal
                    seen_obj = unseen2seen_map[object_name]
                    #seen_obj = 'silver tape' # this is a variant of the experiment
                    lang_goal = pick_google_object.format(seen_obj)

                    # Testing for original goal without modifications
                    #lang_goal = original_lang_goal

                    # define foldername
                    foldername = str(seed) + '_' + object_name.replace(' ','_')
                
                elif eval_task == 'put-block-in-bowl-test-colors':
                    # Loading the success_rate_matrix from the CSV file
                    # success_rate_matrix[place_index][pick_index]
                    success_rate_matrix = np.loadtxt(os.path.join('data', 'success_rate_matrix.csv'), delimiter=",")
                    seen_color = ['red', 'green', 'blue', 'yellow', 'brown', 'gray', 'cyan']
                    unseen_color = ['orange', 'purple', 'pink', 'white']

                    # task goal format 
                    pick_block_in_bowl = 'put the {} blocks in a {} bowl'
                    pick_block_in_towel = 'put the {} blocks in a green and white striped towel'

                    # extract color
                    pick_color = original_lang_goal.split()[2]
                    place_color = original_lang_goal.split()[6]

                    # define foldername
                    foldername = str(seed) + '_' + pick_color + '_' + place_color

                    # Create mappings between color and ID
                    id2color = seen_color + unseen_color
                    color2id = {color: i for i, color in enumerate(id2color)}

                    # map color 
                    # if either pick or place does not exist, then find the corresponding color that maximise the success rate
                    # if both is unseen, then use the combination of green and green
                    if pick_color in unseen_color:
                        if place_color in unseen_color:
                            # both blocks have unseen color
                            pick_color = 'green'
                            place_color = 'green'
                            #print('unseen unseen')
                            lang_goal = pick_block_in_towel.format(pick_color)
                        else:
                            # pick block has unseen color, take the row corresponding to the place_color and pick the color with max value  
                            pick_color = id2color[np.argmax(success_rate_matrix[color2id[place_color]])]
                            #print('unseen seen')
                            lang_goal = pick_block_in_bowl.format(pick_color, place_color)
                    else:
                        if place_color in unseen_color:
                            # place block has unseen color, take the row corresponding to the place_color and pick the color with max value  
                            place_color = id2color[np.argmax(success_rate_matrix[:, color2id[pick_color]])]
                            #print('seen unseen')
                            lang_goal = pick_block_in_towel.format(pick_color)
                        # both blocks have seen color, make no changes 
                        else:
                            #print('seen seen')
                            #lang_goal = pick_block_in_bowl.format(pick_color, place_color)
                            # skip if it is seen x seen colors
                            success_rate_demo_list.append(0)
                            print('seen x seen colors are used in the demo, so success rate should be 0')
                            continue
                            

                    # setup new lang goal
                    #lang_goal = pick_block_in_bowl.format(pick_color, place_color)


                    # change lang goal according to colormap not into green and white striped towel
                    '''
                    # map color 
                    # if either pick or place does not exist, then find the corresponding color that maximise the success rate
                    # if both is unseen, then use the combination of green and green
                    if pick_color in unseen_color:
                        if place_color in unseen_color:
                            # both blocks have unseen color
                            pick_color = 'green'
                            place_color = 'green'
                            #print('unseen unseen')
                            lang_goal = pick_block_in_towel.format(pick_color)
                        else:
                            # pick block has unseen color, take the row corresponding to the place_color and pick the color with max value  
                            pick_color = id2color[np.argmax(success_rate_matrix[color2id[place_color]])]
                            #print('unseen seen')
                            lang_goal = pick_block_in_bowl.format(pick_color, place_color)
                    else:
                        if place_color in unseen_color:
                            # place block has unseen color, take the row corresponding to the place_color and pick the color with max value  
                            place_color = id2color[np.argmax(success_rate_matrix[:, color2id[pick_color]])]
                            #print('seen unseen')
                            lang_goal = pick_block_in_towel.format(pick_color)
                        # both blocks have seen color, make no changes 
                        else:
                            #print('seen seen')
                            #lang_goal = pick_block_in_bowl.format(pick_color, place_color)
                            # skip if it is seen x seen colors
                            success_rate_demo_list.append(0)
                            print('seen x seen colors are used in the demo, so success rate should be 0')
                            continue
                            

                    # setup new lang goal
                    #lang_goal = pick_block_in_bowl.format(pick_color, place_color)
                    '''
            

                # load M edited images for each demo
                edited_images_dir = os.path.join(vcfg['data_dir'], f"{eval_task}-{mode}", 'edited_images')
                edited_images_per_demo = sorted(os.listdir(os.path.join(edited_images_dir, foldername)))
                for j, edited_image in enumerate(edited_images_per_demo):
                    edited_image_path = os.path.join(edited_images_dir, foldername ,edited_image)
                    print(f'Running on edited image {j + 1}/{len(edited_images_per_demo)}, {edited_image_path}')

                    # initialise environment
                    task.mode = mode
                    np.random.seed(seed)
                    env.seed(seed) # set the environment fixed for its seed value
                    env.set_task(task)
                    obs = env.reset(color_change = vcfg['random_BackGroundColor'])
                    info = env.info
                    reward = 0

                    # Format edited image
                    edited_image = Image.open(edited_image_path)
                    edited_image = np.asarray(edited_image)
                    img = torch.from_numpy(ds.process_sample((obs, None, None, info), augment=False)['img'])                       
                    depth = np.array(img.detach().cpu().numpy())[:,:,3]
                    # combine RGB image and depth
                    edited_image = edited_image.transpose(1,0,2) # change back for processing
                    img = np.concatenate((edited_image,
                                            depth[..., np.newaxis],
                                            depth[..., np.newaxis],
                                            depth[..., np.newaxis]), axis=2)


                    # Start recording video (NOTE: super slow)
                    if record:
                        video_name = f'{task_name}-{i+1:06d}'
                        if 'multi' in vcfg['model_task']:
                            video_name = f"{vcfg['model_task']}-{video_name}"
                        env.start_rec(video_name)

                    for _ in range(task.max_steps -1): #TODO edited image is pasted on top so we do not have to run multiple times
                        
                        # update lang goal
                        info['lang_goal'] = lang_goal
                        print(f'Modified Lang Goal: {lang_goal}')
                        
                        # run on edited images, the size of the image needs to be consistent as when the image is saved in demo.py                   
                        act = agent.act(obs, info, goal, edited_image=img)
                        
                        obs, reward, done, info = env.step(act)
                        total_reward += reward
                        # there is a strange error of reward 0 and done is True. Based on affordance map and output, it is found to be an error. Thus, task should be skipped
                        if done is True and reward != 0:
                            total_done += done
                        print(f'Reward: {reward:.3f} | Done: {done}')
                        print(f'Total Reward: {total_reward:.3f} | Total Done: {total_done}\n')
                        if done:
                            break
                    
                    if vcfg['save_affordance']:

                        # get pixel location of pick and place
                        pick, place = act['pick'], act['place']

                        # Create dir and run affordances
                        data_path = os.path.join(vcfg['data_dir'], f"{eval_task}-{mode}")
                        affordance_imgs_dir = os.path.join(data_path, 'affordances', start_datetime)

                        
                        if not os.path.exists(affordance_imgs_dir):
                            os.makedirs(affordance_imgs_dir)

                        affordance_fail_dir = os.path.join(affordance_imgs_dir, 'fail')
                        if not os.path.exists(affordance_fail_dir):
                            os.makedirs(affordance_fail_dir)
                        
                        affordance_success_dir = os.path.join(affordance_imgs_dir, 'success')
                        if not os.path.exists(affordance_success_dir):
                            os.makedirs(affordance_success_dir)

                        # Save affordances
                        if done is True and reward != 0:
                            affordance_path = os.path.join(affordance_imgs_dir, f'success/{seed}_{original_lang_goal}_{lang_goal}_{j+1}.png')
                        else:
                            affordance_path = os.path.join(affordance_imgs_dir, f'fail/{seed}_{original_lang_goal}_{lang_goal}_{j+1}.png')
                            
                        # affordance_path = os.path.join(affordance_imgs_dir, f'{seed}_{obj_name_tosave}.png')

                        run_affordance(img, lang_goal, agent, pick, place, affordance_path, draw_grasp_lines = True, affordance_heatmap_scale=30, alpha_lvl = vcfg['alpha_lvl'])

                
                    # Calculate success rates for demo
                if len(edited_images_per_demo) != 0:
                    success_rate_demo_list.append(total_done / len(edited_images_per_demo))
                    print(f'Success rate for the demo: {success_rate_demo_list[-1]} | Task: {task_name} | Ckpt: {ckpt}')
                else:
                    success_rate_demo_list.append(0)
                    print('No edited images for this demo, so demo success rate is 0')

                # Calculate success rates for task
                success_rate_task = np.mean([r for r in success_rate_demo_list])
                demo_sum = np.sum(success_rate_demo_list)
                print(f'Success rate for the task: {success_rate_task} | Number of demos successful: {demo_sum} | Task: {task_name}')    
                
                # End recording video
                if record:
                    env.end_rec()

            ## end of modification
                
            '''
            all_results[ckpt] = {
                'episodes': results_task,
                'mean_reward': mean_reward_task,
            }
            '''

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


if __name__ == '__main__':
    main()
