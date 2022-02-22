#ifndef GYM_TORCH_H
#define GYM_TORCH_H

#include <torch/torch.h>


class Gym_Torch
{
protected:
    using dType = std::tuple<
                        torch::Tensor,         //Next State
                        torch::Tensor,         //Reward
                        torch::Tensor,         //Done
                        torch::Tensor>;        //Reserved


public:
    virtual torch::Tensor reset() = 0;
    virtual dType step(torch::Tensor action) = 0;
    virtual torch::Tensor sample_action() = 0;
    virtual int action_dimension() = 0;
    virtual int state_dimension() = 0;

    torch::Tensor mState;
};

class CartPole : public Gym_Torch
{
public:
    virtual ~CartPole();

    // Gym_Torch interface
    virtual torch::Tensor reset() override;
    virtual dType step(torch::Tensor action) override;
    virtual torch::Tensor sample_action() override;
    virtual int action_dimension() override;
    virtual int state_dimension() override;

protected:
    const double gravity = 9.8;
    const double masscart = 1.0;
    const double masspole = 0.1;
    const double total_mass = masspole + masscart;
    const double length = 1.0;    // actually half the pole's length
    const double polemass_length = masspole * length;
    const double force_mag = 3*10.0;
    const double tau = 1.0/30.0;  //seconds between state updates 60Hz

    const double theta_threshold_radians = 45.0 * M_PI / 180.0;
    const double x_threshold = 4 * 2.4;

    const std::string kinematics_integrator = "euler";
    int8_t steps_beyond_done = 0;
};

class CartPole_Continous : public CartPole
{
public:
    explicit CartPole_Continous(bool b2D = true);
    virtual ~CartPole_Continous();

    // Gym_Torch interface
    virtual torch::Tensor reset() override;
    virtual dType step(torch::Tensor action) override;
    virtual torch::Tensor sample_action() override;
    virtual int action_dimension() override;
    virtual int state_dimension() override;

protected:
    bool m_b2D;
};

class CartPole_ContinousVision : public CartPole_Continous
{
public:
    explicit CartPole_ContinousVision(bool b2D = true, int preFramesCount = 1);
    virtual ~CartPole_ContinousVision();

    // Gym_Torch interface
    virtual torch::Tensor reset() override;
    virtual dType step(torch::Tensor action) override;
    void setRender_Callback(std::function<std::pair<int,int> (std::vector<double>,
                                                std::vector<double>,
                                                std::vector<unsigned int>&)> *cb);
    // Gym_Torch interface
    int state_dimension() override;

private:
    std::function<std::pair<int,int>(std::vector<double>,
                       std::vector<double>,
                       std::vector<unsigned int>&)> *mRenderCB = nullptr;

    std::deque<torch::Tensor> mvPreFrames;
    int mPreFramesCount = 1;
};

#endif // GYM_TORCH_H
