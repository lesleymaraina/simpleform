class UsersController < ApplicationController
def index
	@users = User.all
end
def new
	@user = User.new
end

def create
	#render text: params.inspect
	@user = User.new(
		params.require(:user).permit(:age, :workclass, :fnlwgt, :education))
	if @user.save
		redirect_to users_url
	else
		render 'new'
	end
end
end



