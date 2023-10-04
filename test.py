import argparse
import torch

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import pre_processing

def get_args():
    # 創建參數解析器
    parser = argparse.ArgumentParser(
        """實現 Deep Q Network 來玩 Flappy Bird""")
    # 定義參數
    parser.add_argument("--image_size", type=int, default=84, help="所有圖片的通用寬度和高度")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args

def test(opt):
    # 檢查是否有 GPU 可用，並設定隨機種子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
        
    # 從已經訓練好的模型中載入神經網絡模型
    if torch.cuda.is_available():
        model = torch.load("{}/flappy_bird".format(opt.saved_path))
    else:
        model = torch.load("{}/flappy_bird".format(opt.saved_path), map_location=lambda storage, loc: storage)
    model.eval()  # 將模型設置為評估模式，不進行梯度更新
    
    # 初始化 Flappy Bird 遊戲狀態
    game_state = FlappyBird()
    
    # 取得遊戲的第一個畫面，並進行預處理
    image, reward, terminal = game_state.next_frame(0)
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)
    image = torch.from_numpy(image)
    
    if torch.cuda.is_available():
        model.cuda() 
        image = image.cuda()
        
    # 將預處理後的畫面堆疊成初始狀態
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]

    while True:
        # 使用模型預測下一個動作
        prediction = model(state)[0]
        
        # 選擇具有最高預測值的動作
        action = torch.argmax(prediction).item()

        # 獲得下一個畫面並進行預處理
        next_image, reward, terminal = game_state.next_frame(action)
        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size,
                                    opt.image_size)
        next_image = torch.from_numpy(next_image)
        
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        
        # 將下一個狀態形成並更新當前狀態
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]
        state = next_state

if __name__ == "__main__":
    opt = get_args()
    test(opt)
