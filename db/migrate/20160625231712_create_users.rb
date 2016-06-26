class CreateUsers < ActiveRecord::Migration
  def change
    create_table :users do |t|
      t.integer :age
      t.string :workclass
      t.integer :fnlwgt
      t.string :education

      t.timestamps null: false
    end
  end
end
